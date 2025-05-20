# Pydantic models for LLM tool inputs and Plan schema
from pydantic import BaseModel, Field
from typing import List, Optional


class GeneratePdfInput(BaseModel):
    markdown_content: str = Field(description="The Markdown content to convert to PDF.")
    filename: str = Field(
        description='The desired filename for the PDF file (e.g., "document.pdf"). Should end with .pdf'
    )


class FetchWebpageContentInput(BaseModel):
    url: str = Field(description="The URL of the webpage to fetch content from.")


class SaveMarkdownInput(BaseModel):
    content: str = Field(description="The markdown content to save.")
    filename: str = Field(
        description='The desired filename for the markdown file (e.g., "summary.md").'
    )


class LaunchTerminalCommandGuiInput(BaseModel):
    command: str = Field(description="The terminal command to launch in a GUI.")
    timeout_seconds: Optional[int] = Field(
        default=30, description="Timeout in seconds for the command execution."
    )


class RunTerminalCommandInput(BaseModel):
    command: str = Field(description="The terminal command to run.")
    timeout_seconds: Optional[int] = Field(
        default=30, description="Timeout in seconds for the command execution."
    )


class PlanStep(BaseModel):
    step: str = Field(description="The name of the step.")
    action: str = Field(description="The command to execute.")
    step_id: str = Field(
        description="A unique identifier for the step, used for tracking and referencing."
    )


class Plan(BaseModel):
    thought: Optional[str] = Field(
        default=None, description="The reasoning behind the generated plan."
    )
    steps: List[PlanStep] = Field(
        description="A list of actionable steps to achieve the user's goal."
    )


class StepOutput(BaseModel):
    name: str
    status: str  # "pending", "running", "complete", "error"
    result: Optional[str] = None
    file_url: Optional[str] = None
    screenshot_url: Optional[str] = None
    error_message: Optional[str] = None
    pdf_url: Optional[str] = None
    plan_step: Optional[PlanStep] = None  # Reference to the original PlanStep


class FinalSummaryResponse(BaseModel):
    summary_intro: str
    original_request: str
    steps: List[StepOutput]
    markdown_url: Optional[str] = None
    accumulated_outputs: Optional[str] = None
    pdf_url: Optional[str] = None
    screenshot_url: Optional[str] = None
