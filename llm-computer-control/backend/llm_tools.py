from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio
from bs4 import BeautifulSoup
import paramiko
import time
from playwright.async_api import async_playwright

from datetime import datetime

load_dotenv()

SSH_HOST = "YOUR_LINUX_VM_IP_OR_DNS" # Provide your Linux VM's public IP or DNS name
SSH_USERNAME = "YOUR_LINUX_VM_USERNAME" # Provide your Linux VM's username
SSH_KEY_PATH = "YOUR_SSH_KEY.pem" # Provide the path to your SSH private key

BACKEND_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# LOCAL_STATIC_DIR is the path where files are saved *locally* by the backend process.
# It should correspond to where FastAPI serves static files from.
# If FastAPI serves /static from project_root/static, and this script is in project_root/backend,
# then LOCAL_STATIC_DIR should point to ../static/summaries
LOCAL_STATIC_SUMMARIES_DIR = os.path.join(
    os.path.dirname(BACKEND_BASE_DIR), "static", "summaries"
)


def _get_ssh_client():
    if not os.path.exists(SSH_KEY_PATH):
        return f"SSH_ERROR: SSH key file not found at {SSH_KEY_PATH}."
    try:
        private_key = paramiko.RSAKey.from_private_key_file(SSH_KEY_PATH)
    except paramiko.ssh_exception.PasswordRequiredException:
        return f"SSH_ERROR: SSH key file at {SSH_KEY_PATH} is encrypted and requires a password."
    except paramiko.ssh_exception.SSHException as e:
        return f"SSH_ERROR: Invalid SSH key file for {SSH_KEY_PATH}. Error: {e}"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            hostname=SSH_HOST, username=SSH_USERNAME, pkey=private_key, timeout=10
        )
        return ssh
    except Exception as e:
        return f"SSH_ERROR: SSH connection to {SSH_HOST} failed: {str(e)}"


# Define path for generated PDFs
LOCAL_STATIC_GENERATED_PDFS_DIR = os.path.join(
    os.path.dirname(BACKEND_BASE_DIR), "static", "generated_pdfs"
)


async def async_generate_pdf_tool(markdown_content: str, filename: str) -> str:
    """Converts the provided Markdown content to a PDF file and saves it.
    The filename should ideally end with .pdf.
    Returns a message with a direct URL to the saved PDF if successful.
    """
    if not markdown_content:
        return "PDF_GENERATION_ERROR: Markdown content to convert is empty or null."

    if not filename.endswith(".pdf"):
        filename += ".pdf"

    # Ensure the local directory for saving generated PDFs exists
    try:
        os.makedirs(LOCAL_STATIC_GENERATED_PDFS_DIR, exist_ok=True)
    except Exception as e:
        return f"PDF_GENERATION_ERROR: Could not create directory {LOCAL_STATIC_GENERATED_PDFS_DIR}: {str(e)}"

    local_file_path = os.path.join(LOCAL_STATIC_GENERATED_PDFS_DIR, filename)

    print(f"Attempting to convert Markdown to PDF and save to: {local_file_path}")

    try:
        # Import locally to ensure it's only used when the tool is called
        from markdown_pdf import MarkdownPdf, Section

        pdf = MarkdownPdf(toc_level=0)  # No ToC for now, can be parameterized later
        # The library expects a list of Section objects or raw markdown strings.
        # Using a single section for simplicity based on current input.
        pdf.add_section(Section(markdown_content, toc=False))
        pdf.save(local_file_path)

        if os.path.exists(local_file_path):
            print(
                f"PDF_GENERATION_SUCCESS: PDF successfully saved to local path: {local_file_path}"
            )
            file_url = f"http://localhost:8000/static/generated_pdfs/{filename}"
            return f"PDF_GENERATION_SUCCESS: PDF generated. Download link: {file_url}"
        else:
            return f"PDF_GENERATION_ERROR: PDF generation failed, file not found at {local_file_path} after save attempt."

    except ImportError:
        return "PDF_GENERATION_ERROR: markdown-pdf library not found. Please ensure it is installed."
    except Exception as e:
        return f"PDF_GENERATION_ERROR: Error converting Markdown to PDF: {str(e)}"


async def fetch_with_http_request(url: str) -> str:
    """Fetches content using standard HTTP request."""
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


# async def async_fetch_webpage_content_tool(url: str, delay: int = 5) -> str:
#     print(f"Attempting to fetch content from URL: {url}")

#     ssh_gui = _get_ssh_client()
#     if isinstance(ssh_gui, paramiko.SSHClient):
#         try:
#             print(f"Opening {url} on remote GUI ({SSH_HOST}) for visual check.")
#             # ssh_gui.exec_command(f'DISPLAY=:1 firefox "{url}" &')
#             ssh_gui.exec_command(
#                 f'DISPLAY=:1 firefox --no-remote --new-instance --profile /tmp/testprofile "{url}" &'
#             )
#             await asyncio.sleep(delay)
#         except Exception as e:
#             print(f"Note: Could not open URL on remote GUI: {str(e)}")
#         finally:
#             ssh_gui.close()
#     else:
#         print(f"Note: Could not connect to remote GUI for visual check: {ssh_gui}")

#     html_content = ""
#     try:
#         async with async_playwright() as p:
#             browser = await p.firefox.launch(headless=True)
#             page = await browser.new_page()
#             stealth_async(page)
#             await page.goto(url, wait_until="domcontentloaded", timeout=45000)
#             html_content = await page.content()
#             await browser.close()
#     except Exception as e:
#         ssh_cleanup = _get_ssh_client()
#         if isinstance(ssh_cleanup, paramiko.SSHClient):
#             try:
#                 ssh_cleanup.exec_command("pkill firefox")
#             except:  # noqa E722
#                 pass
#             finally:
#                 ssh_cleanup.close()
#         return f"FETCH_ERROR: Playwright failed to fetch content from {url}: {str(e)}"

#     ssh_cleanup = _get_ssh_client()
#     if isinstance(ssh_cleanup, paramiko.SSHClient):
#         try:
#             print("Closing Firefox on remote GUI.")
#             ssh_cleanup.exec_command("pkill firefox")
#         except Exception as e:
#             print(f"Note: Could not close Firefox on remote GUI: {str(e)}")
#         finally:
#             ssh_cleanup.close()
#     else:
#         print(f"Note: Could not connect to remote GUI to close Firefox: {ssh_cleanup}")

#     if not html_content:
#         return f"FETCH_ERROR: No HTML content retrieved from {url} by Playwright."

#     print(f"Using BeautifulSoup directly to extract text from {url}.")
#     soup = BeautifulSoup(html_content, "html.parser")
#     if soup.body:
#         extracted_text = soup.body.get_text(separator=" ", strip=True)
#     else:
#         return f"EXTRACT_ERROR: Could not extract any text content from {url} (no body tag after Playwright fetch)."

#     if not extracted_text:
#         return f"EXTRACT_ERROR: No text content could be extracted from {url} after Playwright fetch and parsing."

#     max_length = 20000
#     return f"RAW_CONTENT_SNIPPET: {extracted_text[:max_length]}"


async def async_fetch_webpage_content_tool(url: str, delay: int = 5) -> str:
    print(f"Attempting to fetch content from URL: {url}")
    # Run content extractor script remotely
    ssh_extract = _get_ssh_client()
    if not isinstance(ssh_extract, paramiko.SSHClient):
        return "FETCH_ERROR: Could not connect to remote box for extraction"

    try:
        # Make sure script exists on remote (optional: upload if missing)
        remote_script_path = f"/home/{SSH_USERNAME}/run_content_extractor.sh"
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        remote_filename = f"summary_{timestamp}.md"
        remote_output_path = f"/home/{SSH_USERNAME}/{remote_filename}"

        print(f"Running remote content extractor for: {url}")
        stdin, stdout, stderr = ssh_extract.exec_command(
            f'DISPLAY=:1 {remote_script_path} "{url}" "{remote_filename}"',
        )
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_output = stderr.read().decode("utf-8")
            raise RuntimeError(f"Remote extractor failed: {error_output}")

        # Read extracted content
        sftp = ssh_extract.open_sftp()
        with sftp.open(remote_output_path, "r") as f:
            extracted_text = f.read()

        # Delete the file after reading
        try:
            sftp.remove(remote_output_path)
            print(f"✅ Deleted remote file: {remote_output_path}")
        except Exception as e:
            print(f"⚠️ Could not delete remote file: {e}")
    except Exception as e:
        print(f"⚠️ Could not fetch webpage content: {e}")

        sftp.close()
    finally:
        ssh_extract.close()

    max_length = 20000
    return f"RAW_CONTENT_SNIPPET: {extracted_text[:max_length]}"


async def async_save_markdown_tool(content: str, filename: str) -> str:
    """Saves the provided content to a markdown file on the remote SSH server and downloads it to a local static folder.
    The filename should ideally end with .md.
    Returns a message with a direct URL to the saved file if successful.
    """
    if not content:
        return "SAVE_ERROR: Content to save is empty or null."
    if not filename.endswith(".md"):
        filename += ".md"

    # Ensure the local directory for saving summaries exists
    os.makedirs(LOCAL_STATIC_SUMMARIES_DIR, exist_ok=True)
    local_file_path = os.path.join(LOCAL_STATIC_SUMMARIES_DIR, filename)

    print(
        f"Attempting to save content directly to local markdown file: {local_file_path}"
    )
    try:
        with open(local_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(
            f"SAVE_SUCCESS: Content successfully saved to local path: {local_file_path}"
        )

        # Construct the full URL for the frontend to use
        # This assumes the FastAPI app serves files from a /static route,
        # and LOCAL_STATIC_SUMMARIES_DIR is project_root/static/summaries
        file_url = f"http://localhost:8000/static/summaries/{filename}"
        return f"SAVE_SUCCESS: Content saved. Download link: {file_url}"  # Changed message format slightly for easier parsing

    except Exception as e:
        return f"SAVE_ERROR: Error saving content to local {local_file_path}: {str(e)}"


async def async_launch_terminal_command_gui_tool(
    command: str, timeout_seconds: int
) -> str:
    """Launches a visible terminal in the VNC GUI, runs a command, and returns path to a screenshot."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    remote_filename = f"terminal_{timestamp}.png"
    remote_screenshot_path = f"/home/{SSH_USERNAME}/{remote_filename}"
    # Ensure local_screenshot_path is unique if multiple calls happen, or make it a fixed name.
    # For simplicity, using a fixed name. The FastAPI app might need to handle serving this.
    local_screenshot_dir = os.path.join(
        os.path.dirname(BACKEND_BASE_DIR), "static", "screenshots"
    )
    os.makedirs(local_screenshot_dir, exist_ok=True)
    local_screenshot_path = os.path.join(local_screenshot_dir, remote_filename)
    relative_screenshot_path = f"static/screenshots/{remote_filename}"
    print(f"Launching remote GUI command: {command}")

    ssh = _get_ssh_client()
    if not isinstance(ssh, paramiko.SSHClient):
        return f"GUI_CMD_ERROR: Could not connect to SSH: {ssh}"

    try:
        launch_cmd = f'DISPLAY=:1 xterm -geometry 80x24+100+100 -e "{command}; echo \\$?; echo Command finished. Press Enter or close window.; read" &'
        ssh.exec_command(launch_cmd, timeout=timeout_seconds)
        await asyncio.sleep(4)

        ssh.exec_command(f"DISPLAY=:1 scrot -u {remote_screenshot_path}")
        await asyncio.sleep(2)

        sftp = ssh.open_sftp()
        sftp.get(remote_screenshot_path, local_screenshot_path)
        try:
            sftp.remove(remote_screenshot_path)
            print(f"✅ Deleted remote file: {remote_screenshot_path}")
        except Exception as e:
            print(f"⚠️ Could not delete remote file: {e}")
        sftp.close()

        if os.path.exists(local_screenshot_path):
            file_url = (
                f"http://localhost:8000/{relative_screenshot_path.replace(os.sep, '/')}"
            )
            return f'GUI_CMD_SUCCESS: Command "{command}" launched. Screenshot at {file_url}'
        else:
            return f'GUI_CMD_WARNING: Command "{command}" launched, but screenshot retrieval failed.'
    except Exception as e:
        return f'GUI_CMD_ERROR: Error launching GUI command "{command}": {str(e)}'
    finally:
        if ssh:
            ssh.close()


async def async_run_terminal_command_tool(command: str, timeout_seconds: int) -> str:
    """Runs a terminal command on the remote Linux machine via SSH and returns its stdout/stderr output."""
    print(f"Executing remote non-GUI command: {command}")
    ssh = _get_ssh_client()
    if not isinstance(ssh, paramiko.SSHClient):
        return f"CMD_ERROR: Could not connect to SSH: {ssh}"

    try:
        stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout_seconds)
        output = stdout.read().decode(errors="replace")
        error_output = stderr.read().decode(errors="replace")

        full_result = ""
        if output:
            full_result += f"STDOUT:\n{output}\n"
        if error_output:
            full_result += f"STDERR:\n{error_output}\n"
        if not full_result:
            full_result = "Command produced no output."

        return f'CMD_OUTPUT: Command "{command}" executed.\n{full_result}'
    except Exception as e:
        return f'CMD_ERROR: Error executing command "{command}": {str(e)}'
    finally:
        if ssh:
            ssh.close()
