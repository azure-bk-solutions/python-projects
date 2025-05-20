"""
Langgraph-based backend logic with Chain-of-Thought (CoT) - Structured Output Refactor
Version 9: Implements step_id-based parallel execution with PlanStep in StepOutput and fixes UI update issues
"""

import os
import json
import re
import uuid
from typing import Annotated, List, Dict, Any, Optional, AsyncGenerator
from typing_extensions import TypedDict
import asyncio
import traceback

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import Request

from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    ToolMessage,
    AIMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_openai import ChatOpenAI

from langchain.tools import StructuredTool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from llm_tools import (
    async_fetch_webpage_content_tool,
    async_save_markdown_tool,
    async_launch_terminal_command_gui_tool,
    async_run_terminal_command_tool,
    async_generate_pdf_tool,
)
from llm_schema import (
    FetchWebpageContentInput,
    SaveMarkdownInput,
    LaunchTerminalCommandGuiInput,
    RunTerminalCommandInput,
    GeneratePdfInput,
    PlanStep,  # Updated to use PlanStep with step_id
    Plan,  # Updated Plan model with PlanStep
    StepOutput,
    FinalSummaryResponse,
)

load_dotenv()

MAX_ERRORS_ALLOWED = 3


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    original_request: str
    plan: Optional[Plan]  # Uses the Plan model with PlanStep
    plan_generated: bool  # To track if planner_node has run and set the plan
    current_step_id: str  # Changed from index to step_id
    step_outputs: Dict[str, StepOutput]  # Changed from List to Dict keyed by step_id
    step_order: List[str]  # To maintain step order for UI display
    completed_steps: List[str]  # To track which steps have been completed
    error_count: int
    max_errors: int
    final_summary_response: Optional[Dict[str, Any]]


fetch_content_tool = StructuredTool.from_function(
    coroutine=async_fetch_webpage_content_tool,
    name="fetch_webpage_content",
    description="Fetches raw textual content from a webpage URL.",
    args_schema=FetchWebpageContentInput,
)
save_markdown_tool = StructuredTool.from_function(
    coroutine=async_save_markdown_tool,
    name="save_content_to_markdown",
    description="Saves the provided text content to a markdown file and returns a message with a direct URL to the saved file.",
    args_schema=SaveMarkdownInput,
)
launch_gui_command_tool = StructuredTool.from_function(
    coroutine=async_launch_terminal_command_gui_tool,
    name="launch_terminal_command_gui",
    description="Launches a command in a visible GUI terminal and provides a screenshot URL.",
    args_schema=LaunchTerminalCommandGuiInput,
)
run_terminal_command_tool = StructuredTool.from_function(
    coroutine=async_run_terminal_command_tool,
    name="run_terminal_command",
    description="Runs a non-GUI terminal command and returns its output.",
    args_schema=RunTerminalCommandInput,
)
generate_pdf_tool = StructuredTool.from_function(
    coroutine=async_generate_pdf_tool,
    name="generate_pdf_from_markdown",
    description="Converts the provided Markdown content to a PDF file and returns a message with a direct URL to the saved PDF.",
    args_schema=GeneratePdfInput,
)

cot_tools_list = [
    fetch_content_tool,
    save_markdown_tool,
    launch_gui_command_tool,
    run_terminal_command_tool,
    generate_pdf_tool,
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)
# The Plan model from llm_schema.py is used here for structured output
structured_llm_planner = llm.with_structured_output(Plan)
llm_with_tools = llm.bind_tools(
    cot_tools_list, parallel_tool_calls=True
)  # Enable parallel execution
final_response_llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)


async def initialize_state_node(state: AgentState) -> Dict[str, Any]:
    print("--- INITIALIZING STATE NODE (Structured) ---")
    original_request = state["messages"][0].content
    return {
        "original_request": original_request,
        "plan_generated": False,
        "plan": None,
        "current_step_id": "",  # Will be set when plan is generated
        "step_outputs": {},  # Dictionary keyed by step_id
        "step_order": [],  # To maintain step order
        "completed_steps": [],  # To track completed steps
        "error_count": 0,
        "max_errors": MAX_ERRORS_ALLOWED,
        "final_summary_response": None,
    }


async def planner_node(state: AgentState) -> Dict[str, Any]:
    print("--- PLANNER NODE (Structured) ---")
    original_request = state["original_request"]
    available_tools_description = "\nAvailable Tools:\n"
    for tool in cot_tools_list:
        available_tools_description += f"- {tool.name}: {tool.description}\n"
        if tool.args_schema:
            args_desc = []
            # Use model_json_schema() for Pydantic v2
            schema_props = tool.args_schema.model_json_schema().get("properties", {})
            for field_name, field_props in schema_props.items():
                args_desc.append(
                    f"  - {field_name} ({field_props.get('type', 'N/A')} - {field_props.get('format', 'N/A')} ): {field_props.get('description', 'No description')}"
                )
            if args_desc:
                available_tools_description += "  Args:\n" + "\n".join(args_desc) + "\n"

    planner_system_prompt = (
        "You are an expert planner. Your task is to create a detailed, step-by-step plan "
        "to address the user's request. Each step should be an actionable instruction. "
        "You MUST ONLY use the tools explicitly listed below if a tool is needed for a step. "
        'Do not suggest using other tools or manual operations like "curl" or "opening a terminal manually".'
        f"{available_tools_description}\n"
        "When a step involves using a tool, clearly state the tool name and the arguments you would pass to it, based on its description. "
        "If a step involves fetching content and then saving it, these should be separate steps using the appropriate tools. "
        "If the user asks to save something as a PDF, you should first save it as markdown, then use the 'generate_pdf_from_markdown' tool on that markdown content."
        "Think step by step to create the plan. First, write a draft answer to the query: "
        "Draft answer: ..."
        "Then, make a step by step plan to get to this answer, ensuring each step is either a direct action or a call to one of the available tools."
        "Each step **must include**:"
        " - `step_id`: Unique step number use uuid.uuid4() for this"
        " - `step`: What this step accomplishes - description of this action/step"
        " - `action`: tool to use for this step"
    )
    messages_for_planner = [
        SystemMessage(content=planner_system_prompt),
        HumanMessage(content=f"User request: {original_request}"),
    ]
    initial_step_outputs = {}
    step_order = []
    try:
        # plan_model will be an instance of Plan with PlanStep objects
        plan_model = await structured_llm_planner.ainvoke(messages_for_planner)
        if not isinstance(plan_model, Plan) or not plan_model.steps:
            print(
                "Planner LLM did not return a valid Plan object or the plan has no steps. Falling back."
            )
            # Create a fallback plan with a single step
            fallback_step_id = str(uuid.uuid4())
            fallback_step = PlanStep(
                step=original_request, action="", step_id=fallback_step_id
            )
            plan_model = Plan(
                steps=[fallback_step],
                thought="Fallback: Treating original request as a single step.",
            )
            plan_announcement_content = (
                f"I will address your request directly: {original_request}"
            )
        else:
            print(f"Generated plan: {plan_model.steps}")
            plan_announcement_content = (
                "I have generated the following plan to address your request:\n"
                + "\n".join(
                    [f"{i + 1}. {s.step}" for i, s in enumerate(plan_model.steps)]
                )
            )
            if plan_model.thought:
                plan_announcement_content += f"\n\nMy reasoning: {plan_model.thought}"

        # Initialize step outputs and order
        for i, planstep in enumerate(plan_model.steps):
            step_id = planstep.step_id
            step_order.append(step_id)
            initial_step_outputs[step_id] = StepOutput(
                name=planstep.step,
                status="pending",
                plan_step=planstep,  # Store the entire PlanStep object
            )

        # Set the first step as current if there are steps
        current_step_id = step_order[0] if step_order else ""

        plan_announcement_message = AIMessage(content=plan_announcement_content)
        return {
            "plan": plan_model,  # Storing the Plan model instance in the state
            "plan_generated": True,
            "messages": add_messages(state["messages"], [plan_announcement_message]),
            "step_outputs": initial_step_outputs,
            "step_order": step_order,
            "current_step_id": current_step_id,
        }
    except Exception as e:
        print(f"Error in planner_node: {e}. Original request: {original_request}")
        error_message = AIMessage(
            content=f"I encountered an issue generating a plan: {e}. I will attempt to address your request directly."
        )
        # Create a fallback plan with a single step
        fallback_step_id = str(uuid.uuid4())
        fallback_step = PlanStep(
            step=original_request, action="", step_id=fallback_step_id
        )
        fallback_plan = Plan(
            steps=[fallback_step], thought=f"Fallback due to planner error: {e}"
        )
        step_order = [fallback_step_id]
        initial_step_outputs[fallback_step_id] = StepOutput(
            name=original_request,
            status="pending",
            plan_step=fallback_step,  # Store the entire PlanStep object
        )
        return {
            "plan": fallback_plan,
            "plan_generated": True,
            "messages": add_messages(state["messages"], [error_message]),
            "step_outputs": initial_step_outputs,
            "step_order": step_order,
            "current_step_id": fallback_step_id,
        }


async def execute_step_node(state: AgentState) -> Dict[str, Any]:
    current_step_id = state["current_step_id"]
    step_outputs = state["step_outputs"]
    step_order = state["step_order"]
    completed_steps = state["completed_steps"]

    # Find the next uncompleted step
    next_step_id = None
    for step_id in step_order:
        if step_id not in completed_steps and step_outputs[step_id].status != "running":
            next_step_id = step_id
            break

    # If no uncompleted step found, we're done
    if next_step_id is None:
        print("No more steps to execute")
        return {}

    # Update the current step
    current_step_id = next_step_id
    current_step = step_outputs[current_step_id]
    current_step.status = "running"

    # Get the step instruction from the plan_step stored in StepOutput
    current_step_instruction = (
        current_step.plan_step.step if current_step.plan_step else current_step.name
    )

    # Find the step index for UI display
    step_index = (
        step_order.index(current_step_id) if current_step_id in step_order else -1
    )

    print(
        f"--- EXECUTE STEP NODE (Structured - Step {step_index + 1}/{len(step_order)}: {current_step_instruction}) ---"
    )

    # Prepare previous outputs for context
    previous_outputs_str_list = []
    for step_id in step_order:
        if step_id in completed_steps:
            so = step_outputs[step_id]
            prev_output_entry = (
                f'Output of step "{so.name}":\n{so.result or "No specific result."}'
            )
            if so.file_url:
                prev_output_entry += f"\nFile saved: {so.file_url}"
            if so.pdf_url:
                prev_output_entry += f"\nPDF generated: {so.pdf_url}"
            if so.screenshot_url:
                prev_output_entry += f"\nScreenshot: {so.screenshot_url}"
            previous_outputs_str_list.append(prev_output_entry)

    previous_outputs_str = (
        "\n".join(previous_outputs_str_list) if previous_outputs_str_list else "None"
    )

    # Create the step execution prompt
    step_execution_prompt = (
        f'Original user request: "{state["original_request"]}"\n'
        f"Current plan: {[ps.step for ps in state['plan'].steps]}\n"
        f'We are executing step: "{current_step_instruction}"\n'
        f"Previous step outputs (use this for inputs to the current step if needed, especially if the current step instruction contains placeholders like '[Fetched Content]' or refers to a previously saved markdown file):\n{previous_outputs_str}\n\n"
        "Think step-by-step to execute this current plan step. "
        "If the current step is `save_content_to_markdown` and its `content` argument in the plan is a placeholder like '[Fetched Content]', "
        "you MUST replace that placeholder with the actual text content from the 'Previous step outputs'. "
        "If the current step is `generate_pdf_from_markdown` and its `markdown_content` argument in the plan is a placeholder like '[Content of summary.md]', "
        "you MUST find the actual content of 'summary.md' from the 'Previous step outputs' (likely from a 'save_content_to_markdown' step) and use that as the input. "
        "If the plan refers to a filename for the PDF tool, ensure the `filename` argument for the tool call includes the `.pdf` extension."
        "Restate your goal, consider tools and arguments, explain reasoning, then make the tool call or provide the direct answer."
        f"\nCurrent step ID: {current_step_id}"  # Include step ID for tracking
    )

    messages_for_llm_step = state["messages"] + [
        HumanMessage(content=step_execution_prompt)
    ]

    response = await llm_with_tools.ainvoke(messages_for_llm_step)

    # Update state with the step ID for tracking in process_step_result
    return {
        "messages": add_messages(
            state["messages"],
            [
                HumanMessage(
                    content=f"[Executing Step: {current_step_instruction}] (ID: {current_step_id})",
                    name="system_step_marker",
                ),
                response,
            ],
        ),
        "current_step_id": current_step_id,  # Ensure the current step ID is passed along
        "step_outputs": step_outputs,  # Pass updated step_outputs with running status
    }


cot_tool_node = ToolNode(cot_tools_list)


async def process_step_result_node(state: AgentState) -> Dict[str, Any]:
    current_step_id = state.get("current_step_id", "")
    step_outputs = state.get("step_outputs", {})
    completed_steps = list(state.get("completed_steps", []))
    step_order = state.get("step_order", [])

    if not current_step_id or current_step_id not in step_outputs:
        print(f"Error: Invalid current_step_id {current_step_id}")
        return {}

    # Find the step index for UI display
    step_index = (
        step_order.index(current_step_id) if current_step_id in step_order else -1
    )

    current_step = step_outputs[current_step_id]
    print(
        f"--- PROCESS STEP RESULT NODE (Structured - for step {step_index + 1}, ID: {current_step_id}) ---"
    )

    incoming_error_count = state.get("error_count", 0)

    # Process the result
    last_message = state["messages"][-1]
    step_raw_result_content = "No specific output recorded."
    tool_error_detected = False
    extracted_file_url = None
    extracted_screenshot_url = None
    extracted_pdf_url = None

    if isinstance(last_message, ToolMessage):
        step_raw_result_content = str(last_message.content)
        print(
            f"Processing ToolMessage: {last_message.name}, Content snippet: {step_raw_result_content[:200]}..."
        )
        if any(
            err_tag in step_raw_result_content.upper()
            for err_tag in [
                "ERROR:",
                "FETCH_ERROR",
                "SAVE_ERROR",
                "CMD_ERROR",
                "GUI_CMD_ERROR",
                "PDF_GENERATION_ERROR",
            ]
        ):
            tool_error_detected = True
        if (
            last_message.name == "save_content_to_markdown"
            and "SAVE_SUCCESS:" in step_raw_result_content.upper()
            and "Download link:" in step_raw_result_content
        ):
            url_match = re.search(
                r"Download link:\s*(https?://[\S]+)", step_raw_result_content
            )
            if url_match:
                extracted_file_url = url_match.group(1).strip()
        elif (
            last_message.name == "generate_pdf_from_markdown"
            and "PDF_GENERATION_SUCCESS:" in step_raw_result_content.upper()
            and "Download link:" in step_raw_result_content
        ):
            url_match = re.search(
                r"Download link:\s*(https?://[\S]+)", step_raw_result_content
            )
            if url_match:
                extracted_pdf_url = url_match.group(1).strip()
        elif (
            last_message.name == "launch_terminal_command_gui"
            and "GUI_CMD_SUCCESS:" in step_raw_result_content.upper()
            and "Screenshot at" in step_raw_result_content
        ):
            url_match = re.search(
                r"Screenshot at\s*(https?://[\S]+)", step_raw_result_content
            )
            if url_match:
                extracted_screenshot_url = url_match.group(1).strip()
    elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
        step_raw_result_content = str(last_message.content)
    elif isinstance(last_message, AIMessage) and last_message.tool_calls:
        step_raw_result_content = f"Tool call requested: {last_message.tool_calls}"

    # Update step output and state
    updated_fields = {}

    # Update the current step with results
    current_step = step_outputs[current_step_id]
    current_step.result = step_raw_result_content

    if tool_error_detected:
        updated_fields["error_count"] = incoming_error_count + 1
        current_step.status = "error"
        current_step.error_message = step_raw_result_content
    else:
        current_step.status = "complete"
        if extracted_file_url:
            current_step.file_url = extracted_file_url
        if extracted_screenshot_url:
            current_step.screenshot_url = extracted_screenshot_url
        if extracted_pdf_url:
            current_step.pdf_url = extracted_pdf_url

        # Add to completed steps
        if current_step_id not in completed_steps:
            completed_steps.append(current_step_id)

    # Update the state
    updated_fields["step_outputs"] = step_outputs
    updated_fields["completed_steps"] = completed_steps

    return updated_fields


async def final_summary_node(state: AgentState) -> Dict[str, Any]:
    print("--- FINAL SUMMARY NODE (Structured) ---")
    step_outputs = state.get("step_outputs", {})
    step_order = state.get("step_order", [])
    error_count = state.get("error_count", 0)
    max_errors = state.get("max_errors", MAX_ERRORS_ALLOWED)

    overall_status_intro = "All planned steps have been executed."
    if error_count >= max_errors:
        overall_status_intro = f"Execution stopped due to reaching the maximum error limit ({max_errors} errors)."
    elif error_count > 0:
        overall_status_intro = (
            f"Execution completed, but with {error_count} error(s) encountered."
        )

    step_summary_for_llm = []
    final_markdown_url_from_steps = None
    final_pdf_url_from_steps = None
    final_screenshot_url_from_steps = None

    # Process steps in order
    for i, step_id in enumerate(step_order):
        if step_id in step_outputs:
            so = step_outputs[step_id]
            step_info = f'Step {i + 1} ("{so.name}"): Status - {so.status}.'
            if so.result:
                step_info += f" Result: {so.result[:200]}..."
            if so.file_url:
                print(
                    f"Final summary: md file URL found in step {i + 1}: {so.file_url}"
                )
                step_info += f" File: {so.file_url}"
                final_markdown_url_from_steps = so.file_url
            if so.pdf_url:
                print(f"Final summary: PDF URL found in step {i + 1}: {so.pdf_url}")
                step_info += f" PDF: {so.pdf_url}"
                final_pdf_url_from_steps = so.pdf_url
            if so.screenshot_url:
                print(
                    f"Final summary: screenshot URL found in step {i + 1}: {so.screenshot_url}"
                )
                step_info += f" Screenshot: {so.screenshot_url}"
                final_screenshot_url_from_steps = so.screenshot_url
            if so.error_message:
                step_info += f" Error: {so.error_message[:200]}..."
            step_summary_for_llm.append(step_info)

    llm_prompt_for_summary_intro = (
        f"{overall_status_intro}\n\n"
        f'Original user request: "{state["original_request"]}"\n'
        f"Summary of executed steps:\n" + "\n".join(step_summary_for_llm) + "\n\n"
        "Please provide a concise final summary introduction for the user, highlighting key outcomes and mentioning any important files or errors."
    )

    summary_intro_message = await final_response_llm.ainvoke(
        [HumanMessage(content=llm_prompt_for_summary_intro)]
    )

    generated_summary_intro = summary_intro_message.content

    # Convert step outputs dict to list for FinalSummaryResponse
    step_outputs_list = []
    for i, step_id in enumerate(step_order):
        if step_id in step_outputs:
            step_output = step_outputs[step_id]
            # Create a copy of the StepOutput without the plan_step field for serialization
            step_output_dict = {
                "name": step_output.name,
                "status": step_output.status,
                "result": step_output.result,
                "file_url": step_output.file_url,
                "screenshot_url": step_output.screenshot_url,
                "error_message": step_output.error_message,
                "pdf_url": step_output.pdf_url,
            }
            step_outputs_list.append(step_output_dict)

    accumulated_outputs_str = "\n".join(
        [s.result for s in step_outputs.values() if s.result and s.status == "complete"]
    )

    final_response_data = FinalSummaryResponse(
        summary_intro=generated_summary_intro,
        original_request=state["original_request"],
        steps=step_outputs_list,  # Convert to list for compatibility
        markdown_url=final_markdown_url_from_steps,
        accumulated_outputs=accumulated_outputs_str,
        pdf_url=final_pdf_url_from_steps,
        screenshot_url=final_screenshot_url_from_steps,
    )

    return {
        "messages": add_messages(
            state["messages"], [AIMessage(content=generated_summary_intro)]
        ),
        "final_summary_response": final_response_data.model_dump(),
    }


def should_continue_or_end_after_planning(state: AgentState) -> str:
    if (
        not state.get("plan_generated")
        or not state.get("plan")
        or not state["plan"].steps
    ):
        print(
            "DEBUG should_continue_or_end_after_planning: No plan or empty plan, ending."
        )
        return END  # End if no plan or empty plan
    print(
        "DEBUG should_continue_or_end_after_planning: Plan generated, continuing to execute_step."
    )
    return "continue"


def should_call_tool_or_proceed(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(
            "DEBUG should_call_tool_or_proceed: Tool call detected, proceeding to process_tool_results."
        )
        return "call_tool"
    print(
        "DEBUG should_call_tool_or_proceed: No tool call, proceeding to process_step_result (likely direct LLM response or error)."
    )
    return "proceed_to_process"  # Should go to process_step_result to mark step done or handle LLM direct answer


def should_continue_or_end_after_step(state: AgentState) -> str:
    completed_steps = state.get("completed_steps", [])
    step_order = state.get("step_order", [])
    error_count = state.get("error_count", 0)
    max_errors = state.get("max_errors", MAX_ERRORS_ALLOWED)

    if error_count >= max_errors:
        print(
            f"DEBUG should_continue_or_end_after_step: Max errors ({error_count}) reached. Ending processing."
        )
        return "end_processing"

    # Check if all steps are completed
    all_steps_completed = all(step_id in completed_steps for step_id in step_order)

    if all_steps_completed:
        print(
            f"DEBUG should_continue_or_end_after_step: All steps completed. Ending processing."
        )
        return "end_processing"

    print(
        f"DEBUG should_continue_or_end_after_step: Continuing to next step. Completed: {len(completed_steps)}/{len(step_order)}."
    )
    return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("initialize_state", initialize_state_node)
workflow.add_node("planner", planner_node)
workflow.add_node("execute_step", execute_step_node)
workflow.add_node("process_tool_results", cot_tool_node)
workflow.add_node("process_step_result", process_step_result_node)
workflow.add_node("final_summary", final_summary_node)

# Define the edges
workflow.add_edge(START, "initialize_state")
workflow.add_edge("initialize_state", "planner")
workflow.add_conditional_edges(
    "planner",
    should_continue_or_end_after_planning,
    {
        "continue": "execute_step",
        END: END,
    },
)
workflow.add_conditional_edges(
    "execute_step",
    should_call_tool_or_proceed,
    {
        "call_tool": "process_tool_results",
        "proceed_to_process": "process_step_result",
    },
)
workflow.add_edge("process_tool_results", "process_step_result")
workflow.add_conditional_edges(
    "process_step_result",
    should_continue_or_end_after_step,
    {
        "continue": "execute_step",
        "end_processing": "final_summary",
    },
)
workflow.add_edge("final_summary", END)

# Compile the graph with tracing disabled to prevent tracer exceptions
compiled_graph = workflow.compile()


async def invoke_langgraph_cot_workflow(query: str) -> Dict[str, Any]:
    """
    Invoke the langgraph workflow with the given query.
    Returns the final response data.
    """
    config = {
        "recursion_limit": 100,
        "configurable": {
            "thread_id": f"user-thread-{uuid.uuid4()}",  # Generate unique thread ID
        },
    }
    messages = [HumanMessage(content=query)]
    try:
        final_state = await compiled_graph.ainvoke({"messages": messages}, config)
        if "final_summary_response" in final_state:
            print(
                f"Final summary response found in final_state for non-streaming invocation."
            )
            return final_state[
                "final_summary_response"
            ]  # This is already a dict from model_dump()
        else:
            print(
                "Error: final_summary_response not found in final_state for non-streaming invocation."
            )
            return FinalSummaryResponse(
                summary_intro="An unexpected error occurred, and a full summary could not be generated.",
                original_request=query,
                steps=[],
                markdown_url=None,
                accumulated_outputs="Error during processing.",
                pdf_url=None,
                screenshot_url=None,
            ).model_dump()
    except Exception as e:
        print(f"Error in non-streaming invoke_langgraph_cot_workflow: {e}")
        traceback.print_exc()
        return FinalSummaryResponse(
            summary_intro=f"An error occurred: {str(e)}",
            original_request=query,
            steps=[],
            markdown_url=None,
            accumulated_outputs=traceback.format_exc(),
            pdf_url=None,
            screenshot_url=None,
        ).model_dump()


async def invoke_langgraph_cot_workflow_streaming(
    query: str, request: Request
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Invoke the langgraph workflow with the given query and stream the results.
    Yields events for each step of the workflow.
    """
    print(
        "--- INVOKE STREAMING (User Base Code with Patched Real-time Step Events) ---"
    )
    # Generate a unique thread ID for this streaming session
    thread_id = f"user-thread-stream-{uuid.uuid4()}"
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    inputs = {"messages": [HumanMessage(content=query)]}
    # This dictionary helps prevent sending duplicate state updates
    last_yielded_step_data_by_id = {}
    # Map step_id to index for frontend compatibility
    step_id_to_index_map = {}
    # Track which steps have been reported as completed to the frontend
    reported_completed_steps = set()

    try:
        async for event in compiled_graph.astream_events(
            inputs, config=config, version="v1"
        ):
            if await request.is_disconnected():
                print("Client disconnected, stopping stream.")
                break

            event_kind = event["event"]
            node_name = event.get("name")
            event_data = event.get("data", {})

            if event_kind == "on_chain_end" and node_name == "planner":
                node_output = event_data.get("output")
                if (
                    isinstance(node_output, dict)
                    and "step_outputs" in node_output
                    and isinstance(node_output["step_outputs"], dict)
                    and "step_order" in node_output
                ):
                    step_outputs = node_output["step_outputs"]
                    step_order = node_output["step_order"]

                    # Create step_id to index mapping
                    for i, step_id in enumerate(step_order):
                        step_id_to_index_map[step_id] = i

                    plan_steps_data = []
                    for i, step_id in enumerate(step_order):
                        if step_id in step_outputs:
                            step_output = step_outputs[step_id]

                            # Create a serializable dict without plan_step
                            step_data = {
                                "name": step_output.name,
                                "status": step_output.status,
                                "result": step_output.result,
                                "file_url": step_output.file_url,
                                "screenshot_url": step_output.screenshot_url,
                                "error_message": step_output.error_message,
                                "pdf_url": step_output.pdf_url,
                            }

                            plan_steps_data.append(step_data)
                            last_yielded_step_data_by_id[step_id] = step_data

                    yield {
                        "event": "plan_generated",
                        "data": {"steps": plan_steps_data},
                    }

            elif event_kind == "on_chain_start" and node_name == "execute_step":
                # The input to execute_step is the current AgentState
                current_graph_state_at_step_start = event_data.get("input")
                if isinstance(current_graph_state_at_step_start, dict):
                    current_step_id = current_graph_state_at_step_start.get(
                        "current_step_id"
                    )
                    step_outputs = current_graph_state_at_step_start.get("step_outputs")
                    step_order = current_graph_state_at_step_start.get("step_order")

                    if (
                        current_step_id
                        and isinstance(step_outputs, dict)
                        and current_step_id in step_outputs
                    ):
                        step_to_start = step_outputs[current_step_id]

                        # Create a serializable dict without plan_step
                        step_to_start_data = {
                            "name": step_to_start.name,
                            "status": "running",  # Explicitly set status
                            "result": step_to_start.result,
                            "file_url": step_to_start.file_url,
                            "screenshot_url": step_to_start.screenshot_url,
                            "error_message": step_to_start.error_message,
                            "pdf_url": step_to_start.pdf_url,
                        }

                        last_yielded_step_data_by_id[current_step_id] = (
                            step_to_start_data
                        )

                        # Get step index for frontend compatibility
                        step_index = step_id_to_index_map.get(current_step_id, -1)

                        yield {
                            "event": "step_started",
                            "data": step_to_start_data,
                            "step_index": step_index,
                        }

            elif event_kind == "on_chain_end" and node_name == "process_step_result":
                # The output from process_step_result includes updated step_outputs
                node_output = event_data.get("output")
                current_input = event_data.get("input", {})
                current_step_id = current_input.get("current_step_id")

                if isinstance(node_output, dict) and "step_outputs" in node_output:
                    step_outputs = node_output["step_outputs"]
                    completed_steps = node_output.get("completed_steps", [])

                    if current_step_id and current_step_id in step_outputs:
                        step_obj = step_outputs[current_step_id]

                        # Create a serializable dict without plan_step
                        step_data = {
                            "name": step_obj.name,
                            "status": step_obj.status,
                            "result": step_obj.result,
                            "file_url": step_obj.file_url,
                            "screenshot_url": step_obj.screenshot_url,
                            "error_message": step_obj.error_message,
                            "pdf_url": step_obj.pdf_url,
                        }

                        # Get step index for frontend compatibility
                        step_index = step_id_to_index_map.get(current_step_id, -1)

                        # Only send step_completed event if this step is actually complete
                        # and we haven't reported it yet
                        if (
                            step_obj.status == "complete"
                            and current_step_id not in reported_completed_steps
                            and current_step_id in completed_steps
                        ):
                            reported_completed_steps.add(current_step_id)
                            yield {
                                "event": "step_completed",
                                "data": step_data,
                                "step_index": step_index,
                            }
                            last_yielded_step_data_by_id[current_step_id] = step_data

            elif event_kind == "on_chat_model_stream":
                chunk_obj = event_data.get("chunk")
                content_to_send = ""
                if isinstance(chunk_obj, AIMessageChunk):
                    content_to_send = chunk_obj.content

                if content_to_send:  # Ensure content is not empty
                    yield {"event": "llm_chunk", "data": {"content": content_to_send}}

            elif event_kind == "on_chain_end" and node_name == "final_summary":
                node_output = event_data.get("output")
                if (
                    isinstance(node_output, dict)
                    and "final_summary_response" in node_output
                ):
                    final_summary_data = node_output["final_summary_response"]

                    yield {
                        "event": "final_result",
                        "data": final_summary_data,
                    }

    except Exception as e:
        print(f"Error in streaming: {e}")
        traceback.print_exc()
        error_event = {
            "event": "error",
            "data": {"message": str(e), "traceback": traceback.format_exc()},
        }
        yield error_event
    finally:
        print("SSE stream processing finished.")
