import streamlit as st
# from st_copy_button import st_copy_button
import pandas as pd
import numpy as np
import json
import os
import boto3
import re
import ast
import signal
import tempfile
import traceback
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import time
import io
import matplotlib.pyplot as plt
import matplotlib
# import sklearn
import threading
import hmac, uuid
matplotlib.use("Agg")  # headless backend, avoids GUI warnings in Streamlit


# Imports DataProfiler from dataprofiler module
from dataprofiler import DataProfiler
from agents import Qwen, LLMError
from config import VALID_CHART_TYPES, Union, Dict, Any, Optional, Tuple, List
from figure_utils import tidy_figure



@st.cache_resource
def get_canvas_lock():
    return threading.Lock()

canvas_lock = get_canvas_lock()


# ==========================================
# BLOCK 0: CONFIG & PATHS & ACCESS
# ==========================================
st.set_page_config(
    page_title = "AI Data Viz Agent", 
    page_icon = "robotfavicon2.png",
    layout = "wide"
)


DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/data"))
# os.makedirs(DATA_DIR, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_LOG = DATA_DIR / "feedback.jsonl"
RUN_LOG = DATA_DIR / "runs.jsonl"
PROMPT_VERSION = "v2-intent-first"   # bump whenever you edit the Thinker/Coder/Debugger prompts
st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])


def _read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True, dtype={"run_id": str, "session_id": str, "user": str})


# Gatekeeping for access to AI agent
_code = os.getenv("ACCESS_CODE")
if not _code:
    st.error("Server misconfigured."); st.stop()
if not st.session_state.get("ok"):
    entered = st.text_input("Access code", type="password")
    if entered and hmac.compare_digest(entered, _code):
        st.session_state.ok = True; st.rerun()
    st.stop()
    
    
if "user" not in st.session_state:
    name = st.text_input("Your first name")
    if name.strip():
        st.session_state.user = name.strip().lower()[:30]; st.rerun()
    st.stop()
    
    
_admin = os.getenv("ADMIN_CODE")
if _admin:  # no ADMIN_CODE set = admin panel doesn't exist
    with st.sidebar.expander("🔒 Admin"):
        entered = st.text_input("Admin code", type="password", key="admin_code")
        if entered and hmac.compare_digest(entered, _admin):
            runs, fb = _read_log(RUN_LOG), _read_log(FEEDBACK_LOG)
            runs = runs.drop_duplicates("run_id", keep="last")
            
            if runs.empty:
                st.caption("No runs logged yet.")
            else:
                st.metric("Runs", len(runs))
                if "refused" not in runs:
                    runs["refused"] = 0   
                runs["refused"] = runs["refused"].fillna(0)
                valid = runs[runs["refused"] == 0]
                st.metric("Success rate", f"{valid['success'].mean():.0%}" if len(valid) else "n/a")
                st.metric("Refusals", int(runs["refused"].sum()))
                st.metric("Testers", runs["user"].nunique())
                st.download_button("⬇️ mvpruns.csv", runs.to_csv(index=False),
                                   "mvpruns.csv", "text/csv", key="dl_runs")
            if not fb.empty:
                st.download_button("⬇️ feedback.csv", fb.to_csv(index=False),
                                   "feedback.csv", "text/csv", key="dl_fb")
                    

# st.session_state.setdefault("runs", 0)
# # in the run block:
# if st.session_state.runs >= 25:
#     st.warning("Run limit reached for this session."); st.stop()
# st.session_state.runs += 1

    



# # Feedback log lives on Drive if mounted, otherwise in /tmp (ephemeral)
# DRIVE_ROOT = Path("/content/drive/MyDrive")
# if DRIVE_ROOT.exists():
#     FEEDBACK_DIR = DRIVE_ROOT / "viz_agent_feedback"
# else:
#     FEEDBACK_DIR = Path("/tmp/viz_agent_feedback")
# FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
# FEEDBACK_LOG = FEEDBACK_DIR / "feedback.jsonl"



def uploadLogToS3(local_file_path, s3_bucket_name, s3_file_name):
    s3 = boto3.client('s3', region_name='us-east-2')
    
    try:
        # 1. Read local JSONL logs
        if not Path(local_file_path).exists():
            return
        
        local_df = pd.read_json(local_file_path, lines=True, dtype={"run_id": str, "session_id": str, "user": str})
        if local_df.empty:
            return
            
        # 2. Attempt to download existing file from S3 using the variable s3_file_name
        s3_df = pd.DataFrame()
        try:
            response = s3.get_object(Bucket=s3_bucket_name, Key=s3_file_name)  # ✅ Fixed: dynamic key
            s3_df = pd.read_csv(io.BytesIO(response['Body'].read()), dtype={"run_id": str, "session_id": str, "user": str})
        except s3.exceptions.NoSuchKey:
            # ✅ First time creating mvpruns.csv in S3 - continue execution!
            print(f"[S3 Info] {s3_file_name} does not exist in S3 yet. Creating new file.")
        except Exception as e:
            print(f"[S3 Warning] Error fetching {s3_file_name}: {e}")

        # 3. Combine S3 data and local data, deduplicating on run_id
        if not s3_df.empty:
            combined_df = pd.concat([s3_df, local_df], ignore_index=True)
            if 'run_id' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['run_id'], keep='last')
        else:
            combined_df = local_df

        # 4. Write CSV locally and upload to S3
        csv_local_path = '/tmp/runs_output.csv'
        combined_df.to_csv(csv_local_path, index=False)
        
        s3.upload_file(csv_local_path, s3_bucket_name, s3_file_name)
        print(f"[S3 Sync] Successfully updated {s3_file_name} in bucket {s3_bucket_name} with {len(combined_df)} rows.")

    except Exception as e:
        print(f"Failed to upload to S3: {e}", flush=True)


# ==========================================
# BLOCK 4: SAFE EXECUTION
# ==========================================
BANNED_MODULES = {
    'os', 'sys', 'subprocess', 'socket', 'shutil', 'pathlib',
    'importlib', 'ctypes', 'builtins', '__builtin__', 'requests', 'urllib',
    'http', 'ftplib', 'smtplib', 'pickle', 'marshal', 'code',
}
BANNED_NAMES = {
    'eval', 'exec', 'compile', '__import__', 'open',
    'input', 'breakpoint', 'globals', 'vars', 'getattr', 'setattr', 'delattr',
}

BANNED_ATTRS = {
    'os', 'sys', 'environ', 'getenv', 'system', 'popen', 'subprocess',
    'savefig', 'imread', 'imsave', 'urlopen', 'loadtxt', 'savetxt',
    'genfromtxt', 'fromfile', 'tofile',
    'to_csv', 'to_excel', 'to_json', 'to_pickle', 'to_parquet', 'to_sql',
}


def validate_code_safety(code: str) -> Tuple[bool, str]:
    """Static AST check for dangerous patterns. Not a full sandbox — pair with a timeout."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in BANNED_MODULES:
                    return False, f"Banned import: {alias.name}"
        if isinstance(node, ast.While):
            return False, "while loops are not allowed"
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in BANNED_MODULES:
                return False, f"Banned import from: {node.module}"
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return False, f"Banned name reference: {node.id}"
        if isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS or node.attr.startswith('read_'):
                return False, f"Blocked attribute: {node.attr}"
            
            ALLOWED_DUNDERS = {'__init__', '__name__'}
            if node.attr.startswith('__') and node.attr.endswith('__') and node.attr not in ALLOWED_DUNDERS:
                return False, f"Dunder attribute access blocked: {node.attr}"
    return True, "OK"


class ExecTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    """Cross-platform timeout context manager. Falls back to no-op on Windows."""
    if not hasattr(signal, 'SIGALRM') or threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise ExecTimeout(f"Execution exceeded {seconds}s")
    
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def safe_exec(code: str, df: pd.DataFrame, timeout_s: int = 30) -> Tuple[Optional[plt.Figure], Optional[str]]:
    if not code.strip():
        return None, "Coder returned empty code (likely truncated output)."

    ok, reason = validate_code_safety(code)
    if not ok:
        return None, f"Safety check failed: {reason}"
    
    with canvas_lock:
        plt.close('all')
        sandbox_globals = {
            'df': df.copy(), 'plt': plt, 'pd': pd, 'np': np, 'matplotlib': matplotlib,
            # no pre-made 'ax'; the prompt already tells the coder to use plt.gca()/subplots
        }
        try:
            with time_limit(timeout_s):
                exec(code, sandbox_globals)          # ONE dict, not (globals, {})
                
            def data_in_view(ax) -> bool:
                dl = ax.dataLim
                if not np.isfinite(dl.extents).all():
                    return False
                xlo, xhi = sorted(ax.get_xlim()); ylo, yhi = sorted(ax.get_ylim())
                return dl.x1 >= xlo and dl.x0 <= xhi and dl.y1 >= ylo and dl.y0 <= yhi

            def has_content(f):
                return any((a.lines or a.patches or a.collections or a.containers or a.images) and data_in_view(a) for a in f.axes)

            # Pick the newest figure that actually has data, not blindly gcf()
            figs = [plt.figure(n) for n in plt.get_fignums()]
            good = [f for f in figs if has_content(f)]
            if not good:
                return None, "Code ran but produced no plot with data."
            fig = good[-1]

            try:
                tidy_figure(fig)
            except Exception:
                pass
            return fig, None
        except ExecTimeout as e:
            return None, str(e)
        except Exception:
            return None, traceback.format_exc()


# ==========================================
# BLOCK 5: PIPELINE
# ==========================================


def extract_json(text: str) -> Optional[Dict]:
    """Robustly pull a JSON object from LLM output, tolerating prose around it."""
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the largest {...} span
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def validate_plan(plan: Dict, df_columns: List[str]) -> Tuple[bool, str]:
    if not isinstance(plan, dict):
        return False, "Plan is not a dict"
    ct = str(plan.get("chart_type") or "").lower()
    if ct not in VALID_CHART_TYPES:
        return False, f"Invalid chart_type: {ct!r}"
    plan["chart_type"] = ct
    cols = plan.get("relevant_columns")
    if not isinstance(cols, list) or not cols:
        return False, "No relevant_columns specified"
    missing = [c for c in cols if c not in df_columns]
    if missing:
        return False, f"Columns not in dataset: {missing}"
    if not isinstance(plan.get("human_readable_labels"), dict):
        plan["human_readable_labels"] = {}   # the Coder calls .get() on this
    return True, "OK"


def run_pipeline(df: pd.DataFrame, df_summary: str, user_prompt: str,
                 previous_plan: Optional[Dict] = None,
                 previous_code: Optional[str] = None,
                 max_retries: int = 3):
    llm = Qwen()
    st.session_state.llm = llm
    st.session_state.debug_log = []
    st.session_state.refused = False   # reset every run
    st.session_state.failed_stage = ""
    # st.session_state.intent_overridden = False
    

    def log(label, content, level="info"):
        st.session_state.debug_log.append({"label": label, "content": str(content), "level": level})

    # --- 1. Thinker ---
    with st.status("🧠 Thinker agent is running...", expanded=True) as status:
        raw_thinker = ''
        try: 
            raw_thinker = llm.thinker(df_summary, user_prompt, previous_plan, previous_code)
        except Exception as e: 
            status.update(label=f"🧠 Thinker error: {type(e).__name__}", state="error")
            st.error(f"An unexpected error occurred in the Thinker agent: {e}")
            return None, None, None
                   
        log("Thinker raw output", raw_thinker)
        plan_json = extract_json(raw_thinker)
        
        is_refusal = isinstance(plan_json, dict) and (
            plan_json.get("refused")
            or (not str(plan_json.get("chart_type") or "").strip()
                and not plan_json.get("relevant_columns"))
        )
        
        if is_refusal:
            st.session_state.refused = True
            st.session_state.failed_stage = "refused"
            status.update(label="🧠 Request out of scope", state="error")
            st.warning(plan_json.get("message") or
                    "I'm a data visualization agent and can only help create charts from your dataset.")
            return None, None, None

        if plan_json is None:
            st.session_state.failed_stage = "thinker_parse"
            status.update(label="🧠 Thinker: JSON parse failed", state="error")
            st.error("Could not extract valid JSON from Thinker output.")
            st.code(raw_thinker)
            return None, None, None

        
        named = [t for t in VALID_CHART_TYPES if re.search(rf"\b{re.escape(t)}\b", user_prompt.lower())]
        if isinstance(plan_json, dict) and len(named) == 1 and plan_json.get("chart_type") != named[0]:
            log("Intent override", f"{plan_json.get('chart_type')} -> {named[0]}")
            plan_json["chart_type"] = named[0]
            # st.session_state.intent_overridden = True
            
        ok, reason = validate_plan(plan_json, list(df.columns))
            
        if not ok:
            st.session_state.failed_stage = "plan_invalid"
            status.update(label=f"🧠 Thinker: plan invalid — {reason}", state="error")
            st.error(f"Plan validation failed: {reason}")
            st.json(plan_json)
            return None, None, None

        status.update(label=f"🧠 Plan: {plan_json.get('task_summary', '')}",
                      state="complete", expanded=False)
        st.write(f"**Chart type:** `{plan_json.get('chart_type')}` | "
                 f"**Columns:** `{plan_json.get('relevant_columns')}`")
        st.json(plan_json, expanded=False)

    # --- 2. Coder ---
    with st.status("💻 Coder agent is generating code...", expanded=True) as status:
        raw_coder = ''
        try:
            raw_coder = llm.coder(df_summary, plan_json, previous_code)
        except Exception as e:
            st.session_state.failed_stage = "coder_error"
            status.update(label=f"💻 Coder error: {type(e).__name__}", state="error")
            st.error(f"An unexpected error occurred in the Coder agent: {e}")
            return None, None, None
        
        final_code = re.sub(r'^python\s*', '', raw_coder, flags=re.IGNORECASE).strip()
        log("Coder initial output", final_code)
        status.update(label="💻 Code generated!", state="complete", expanded=False)
        st.code(final_code, language="python")

    # --- 3. Execute with debug retries ---
    for attempt in range(max_retries):
        label = f"attempt {attempt + 1}/{max_retries}"
        with st.status(f"⚙️ Executing — {label}...", expanded=True) as status:
            fig, err = safe_exec(final_code, df, timeout_s=30)
            if fig is not None:
                status.update(label=f"✅ Execution succeeded — {label}",
                              state="complete", expanded=False)
                return fig, final_code, plan_json

            log(f"Exec error {label}", err, level="error")
            status.update(label=f"❌ Execution failed — {label}", state="error")
            st.code(err)

        if attempt < max_retries - 1:
            with st.status(f"🔧 Debugger agent is rewriting ({label})...", expanded=True) as dbg:
                raw_debug = ''
                try:
                    raw_debug = llm.debugger(df_summary, plan_json, final_code, err or "")
                except Exception as e:
                    st.session_state.failed_stage = "debugger_error"
                    dbg.update(label="🔧 Debugger failed", state="error")
                    st.error("The model service failed. Please try again.")
                    break

                final_code = re.sub(r'^python\s*', '', raw_debug, flags=re.IGNORECASE).strip()
                log(f"Debugger output {label}", final_code)
                dbg.update(label=f"🔧 Debugger done — {label}", state="complete")
                st.code(final_code, language="python")

    st.session_state.failed_stage = st.session_state.failed_stage or "execution"
    st.error(f"🛑 Max retries reached.")
    return None, final_code, plan_json


# ==========================================
# BLOCK 6: FEEDBACK LOGGING
# ==========================================

def log_run(run_id, prompt, plan, code, ok, seconds, df, attempts, refine = False, source = ""):
    
    try:
        llm = st.session_state.get("llm")
        rec = {
            "date_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": run_id,
            "session_id": st.session_state.sid,
            "user": st.session_state.get("user", ""),
            "success": int(ok),
            "failure_stage": "" if ok else st.session_state.get("failed_stage", ""),
            "time_secs": round(seconds, 1),
            "named_chart_type": int(any(re.search(rf"\b{re.escape(t)}\b", prompt.lower()) for t in VALID_CHART_TYPES)),
            "chart_type": (plan or {}).get("chart_type", ""),
            "num_of_attempts": attempts,
            # "columns_used": ",".join(map(str, (plan or {}).get("relevant_columns", []))),
            # "n_rows": df.shape[0], "n_cols": df.shape[1],
            "refine_chart": int(refine),
            "data_source": "Example" if str(st.session_state.get("df_name", "")).startswith("example:") else "Upload",   # "Upload" or "Example"
            "prompt_len": len(prompt.lower()), 
            "mentions_column": int(any(re.search(rf"\b{re.escape(str(c).lower())}\b", prompt.lower()) for c in df.columns)),
            "refused" : int(st.session_state.get("refused", False)),
            "input_tokens" : llm.total_input_tokens if llm else 0,
            "output_tokens" : llm.total_output_tokens if llm else 0,
            "used_suggested_prompt" : int(st.session_state.get("used_suggestion", False)),
            "prompt_version" : PROMPT_VERSION,
            "rating": None,
            "rated_ts": None,
            # "code": code or "",
        }
        
        if ok:
            st.session_state.last_ok_rec = rec
        
        with open(RUN_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
            
        # 1. Track local run counts in session state
        st.session_state.setdefault("runs_since_last_upload", 0)
        st.session_state.runs_since_last_upload += 1
        
        # 2. Define your batch threshold (e.g., upload every 2 runs)
        BATCH_THRESHOLD = 1 
        
        if st.session_state.runs_since_last_upload >= BATCH_THRESHOLD:
            uploadLogToS3(str(RUN_LOG), "ai-agent-metrics-bucket", "mvpruns.csv")
            # Reset the counter after a successful batch upload
            st.session_state.runs_since_last_upload = 0
            print(f"[S3 Sync] Uploaded batch of {BATCH_THRESHOLD} runs to S3.")
            
    except Exception as e:
        print(f"[log_run failed] {type(e).__name__}: {e}", flush=True)
    

def log_feedback(rating, prompt=None, plan=None, code=None):
    rec = st.session_state.get("last_ok_rec")
    if not rec:
        st.toast("Nothing to rate yet", icon="⚠️")
        return
    rated = {**rec,
             "rating": 1 if rating == "up" else 0,
             "rated_ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        with open(RUN_LOG, "a") as f:
            f.write(json.dumps(rated) + "\n")
        uploadLogToS3(str(RUN_LOG), "ai-agent-metrics-bucket", "mvpruns.csv")
        st.toast("Thanks for the feedback!", icon="✅")
    except Exception as e:
        st.toast(f"Feedback log failed: {e}", icon="⚠️")


# ==========================================
# BLOCK 7: EXAMPLE DATASETS
# ==========================================
def make_example_dataset(name: str) -> Tuple[pd.DataFrame, str]:
    if name == "iris":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame
        df['species'] = df['target'].map(dict(enumerate(data.target_names)))
        df = df.drop(columns=['target'])
        return df, "Iris flower measurements: try 'scatter of sepal length vs petal length colored by species'"
    if name == "sales_timeseries":
        rng = np.random.default_rng(42)
        dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
        trend = np.linspace(100, 250, len(dates))
        season = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        noise = rng.normal(0, 10, len(dates))
        df = pd.DataFrame({
            "date": dates,
            "revenue": trend + season + noise,
            "region": rng.choice(["North", "South", "East", "West"], len(dates)),
        })
        return df, "Daily revenue by region: try 'line chart of revenue over time'"
    if name == "survey":
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            "age": rng.integers(18, 70, n),
            "income_k": rng.normal(60, 20, n).round(1),
            "satisfaction": rng.choice(["Low", "Medium", "High"], n, p=[0.2, 0.5, 0.3]),
            "department": rng.choice(["Eng", "Sales", "HR", "Ops"], n),
        })
        return df, "Employee survey: try 'bar chart of satisfaction counts by department'"
    raise ValueError(name)


# ==========================================
# BLOCK 8: STREAMLIT UI
# ==========================================
st.title("📊 Visualize your data in just a few steps!")
st.info("Upload a dataset, describe the chart you want (BE AS SPECIFIC AS POSSIBLE. e.g. mention chart types for better results), and click 'Run!' Three specialized LLM agents (Thinker → Coder → Debugger) collaborate on every request. Powered by Qwen.")
st.warning("ℹ️ Beta: Specific variables are logged for calculating metrics (e.g. prompt length and success rate), and column names/sample values are sent to a third-party AI model (AI providers may retain requests under their own policies). Please don't upload sensitive data.")


# Session state init
for key, default in [("history", []), ("last_plan", None), ("last_code", None),
                     ("last_fig", None), ("last_prompt", None), ("df", None),
                     ("df_summary", None), ("df_name", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- Helpers used by the main panel ---

def _reset_session_artifacts():
    st.session_state.history = []
    st.session_state.last_plan = None
    st.session_state.last_code = None
    st.session_state.last_fig = None
    st.session_state.last_prompt = None
    st.session_state.last_ok_rec = None


def render_chart_panel(container, fig, plan, code, prompt, turn_id):
    """Render chart + plan summary + download/feedback buttons into the pinned area."""
    with container.container():
        st.pyplot(fig, use_container_width=True)

        summary_col, actions_col = st.columns([3, 2])

        with summary_col:
            st.markdown(f"**📝 Prompt:** _{prompt}_")
            cols_str = ", ".join(f"`{c}`" for c in plan.get("relevant_columns", []))
            st.markdown(f"**📐 Chart:** `{plan.get('chart_type')}` &nbsp;·&nbsp; **Columns:** {cols_str}")
            task = plan.get("task_summary", "")
            if task:
                st.caption(task)

        with actions_col:
            b1, b2, b3 = st.columns(3)
            buf_png = io.BytesIO()
            fig.savefig(buf_png, format="png", dpi=150, bbox_inches="tight")
            b1.download_button("⬇️ Download PNG", buf_png.getvalue(), "plot.png", "image/png",
                               use_container_width=True, key=f"png_{turn_id}")
            buf_svg = io.BytesIO()
            fig.savefig(buf_svg, format="svg", bbox_inches="tight")
            b2.download_button("⬇️ Download SVG", buf_svg.getvalue(), "plot.svg", "image/svg+xml",
                               use_container_width=True, key=f"svg_{turn_id}")
            b3.download_button("⬇️ Download Code", code, "plot.py", "text/plain",
                               use_container_width=True, key=f"code_{turn_id}")

            f1, f2 = st.columns(2)
            if f1.button("👍 Good", use_container_width=True, key=f"up_{turn_id}"):
                log_feedback("up", prompt, plan, code)
            if f2.button("👎 Bad", use_container_width=True, key=f"dn_{turn_id}"):
                log_feedback("down", prompt, plan, code)


def suggest_prompts(df: pd.DataFrame) -> list:
    """Generate 3-4 prompt suggestions from the loaded dataframe's column types."""
    df = df.loc[:, ~df.columns.astype(str).str.startswith(pat = ("Unnamed", "?", "NaN"))]
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = [c for c in df.columns if 'datetime' in str(df[c].dtype)]
    low_card_cat = [c for c in df.columns
                    if (df[c].dtype == 'object' or df[c].dtype.name == 'category')
                    and 1 < df[c].nunique() < 20]
    low_card_int = [c for c in numeric if df[c].nunique() < 20 and df[c].nunique() > 1]

    suggestions = []
    if datetime_cols and numeric:
        suggestions.append(f"line chart of {numeric[0]} over {datetime_cols[0]}")
    if len(numeric) >= 2:
        suggestions.append(f"scatter plot of {numeric[0]} vs {numeric[1]}")
    if (low_card_cat or low_card_int) and numeric:
        grp = (low_card_cat or low_card_int)[0]
        y = next((c for c in numeric if c != grp), numeric[0])
        suggestions.append(f"box plot of {y} by {grp}")
    if len(low_card_cat) >= 2:
        suggestions.append(f"grouped bar chart of counts by {low_card_cat[0]} and {low_card_cat[1]}")
    if numeric:
        suggestions.append(f"line chart of {numeric[-1]}")
    return suggestions[:4]


def use_prompt(text: str):
    st.session_state.user_prompt = text
    st.session_state.used_suggestion = True
    

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Generate a visualization using the following steps! ")

    st.divider()
    st.markdown("**Step 1: Load data**")
    source = st.radio("Dataset source", ["Upload", "Example"],
                      horizontal=True, label_visibility="collapsed")

    df_loaded = False
    if source == "Upload":
        uploaded_file = st.file_uploader(
            "Upload CSV, TXT, XLSX, or JSON",
            type=["csv", "txt", "xlsx", "xls", "json"],
            help="Files above ~100 MB may be slow to profile.",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            fid = f"{uploaded_file.name}:{len(file_bytes)}"
            if st.session_state.df_name != fid:
                ext = Path(uploaded_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                profiler = DataProfiler(tmp_path)
                try:
                    df = profiler.load_data()
                    st.session_state.df = df
                    st.session_state.df_summary = profiler.compact_summary()
                    st.session_state.df_name = fid
                    _reset_session_artifacts()
                except Exception as e:
                    st.error(f"Failed to load: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            df_loaded = st.session_state.df is not None
    else:
        example = st.selectbox(
            "Example dataset",
            ["iris", "sales_timeseries", "survey"],
            help="iris: 3 species × 4 measurements. sales_timeseries: daily revenue by region. survey: employee satisfaction.",
        )
        if st.button("Load example", use_container_width=True):
            df, hint = make_example_dataset(example)
            st.session_state.df = df
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                df.to_csv(tmp.name, index=False)
                tmp_path = tmp.name
            profiler = DataProfiler(tmp_path)
            profiler.load_data()
            st.session_state.df_summary = profiler.compact_summary()
            st.session_state.df_name = f"example:{example}"
            _reset_session_artifacts()
            st.toast(hint, icon="💡")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        df_loaded = st.session_state.df is not None


    st.divider()
    st.markdown("**Step 2: Describe your chart**")
    # Expandable suggested prompts section
    with st.expander("📝 Suggested prompts"):
        if st.session_state.df is None:
            st.caption("Load a dataset in Step 1 to see suggestions.")
        else:
            suggestions = suggest_prompts(st.session_state.df)
            if suggestions:
                st.markdown("**Click a prompt and autopopulate the text box:**")
                for i, s in enumerate(suggestions):
                    # st.markdown(f"- `{s}`")
                    # st.button(s, key = '▸', use_container_width=True, on_click=lambda s=s: st.session_state.update(user_prompt=s))
                    # st_copy_button(text = s, before_copy_label = f'▸ {s}', after_copy_label = 'Prompt copied ✅', show_text = False) - Doesn't work on deployed web app
                    st.button(
                        f"▸ {s}",
                        key=f"sugg_{i}",
                        use_container_width=True,
                        on_click=use_prompt,
                        args=(s,),
                    )
                st.caption("You can also modify the given prompt, or craft your own 😊.")
                
    user_prompt = st.text_area(
        "Your request",
        placeholder="e.g. line chart of revenue over time",
        height=100,
        key="user_prompt",
        label_visibility="collapsed",
        help="Include the chart type and column names when you can. The Thinker picks the right chart based on what you say and what your data looks like.",
    )

    refine = False
    if st.session_state.last_plan is not None:
        refine = st.checkbox(
            "🔁 Refine previous chart",
            value=False,
            help="Feeds the last plan + code to the Thinker so follow-up tweaks like 'make the bars red' work without restarting.",
        )

    st.divider()
    st.sidebar.markdown("**Step 3: Run!**")
    
    # 1. Wrap your Run button inside a uniquely keyed container
    run_button = st.sidebar.button(
        "🚀 Run!",
        use_container_width=True,
        disabled=not df_loaded or not user_prompt.strip(),
        type="primary",
    )
    
    # 2. Inject CSS to color it green with a smooth hover effect
    st.sidebar.markdown("""
        <style>
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
        [data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
            background-color: #2e7d32 !important;
            border: 1px solid #2e7d32 !important;
            color: white !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover,
        [data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {
            background-color: #388e3c !important;
            border-color: #388e3c !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    
    MAX_RUNS = 25
    st.session_state.setdefault("runs", 0)
    DAILY_CAP = int(os.getenv("DAILY_CAP", "150"))
    def runs_today() -> int:
        # if not RUN_LOG.exists(): return 0
        # today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # with open(RUN_LOG) as f:
        #     return sum(1 for line in f if f'"ts": "{today}' in line)
        
        # New code:
        if not RUN_LOG.exists(): return 0
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ids = set()
        with open(RUN_LOG) as f:
            for line in f:
                if f'"date_time": "{today}' in line:
                    try: ids.add(json.loads(line)["run_id"])
                    except Exception: pass
        return len(ids)

    over_limit = st.session_state.runs >= MAX_RUNS or runs_today() >= DAILY_CAP


    if run_button and over_limit:
        st.warning("Run limit reached for this session.")

    if run_button and not over_limit and st.session_state.df is not None and user_prompt.strip():
        st.session_state.runs += 1
    
    if not df_loaded:
        st.caption("_Load data first to enable Run._")
    elif not user_prompt.strip():
        st.caption("_Enter a prompt in Step 2 to enable Run._")

    if st.session_state.history:
        if st.button("🧹 Clear history", use_container_width=True):
            _reset_session_artifacts()
            st.rerun()

    # st.divider()
    # with st.expander("ℹ️ Instructions", expanded=False):
    #     st.markdown("""
    #     **Workflow**
    #     1. **Load data** — upload a file or pick an example.
    #     2. **Describe** the chart — mention chart type and column names if you know them. Be as specific as possible.
    #     3. **Run** — the agent plans, writes code, and executes it (with up to 3 debug retries).
    #     4. **Refine** — tick 🔁 Refine and send a follow-up like "log-scale the y-axis".

    #     **Good prompts**
    #     - `line chart of revenue over time`
    #     - `scatter plot of height vs weight`
    #     - `box plot of exam_score by study_hours`
    #     - `distribution of income`
    #     - `grouped bar chart of counts by department and satisfaction`

    #     **Notes**
    #     - Generated code is AST-checked
    #     - Downloads (PNG, SVG, Python code) appear with each successful run.
    #     - 👍 / 👎 feedback is logged for evaluation.
    #     """)


# === MAIN PANEL ===

# Pinned chart display area — reserved at the TOP so new charts always surface here.
chart_area = st.empty()


def render_welcome(container):
    with container.container():
        st.markdown("### Welcome! 👋")
        st.markdown(
            "This AI tool turns a plain-English description into a publication-ready chart. "
            "Pick or upload your own dataset from the sidebar, tell the agent what you want to see, and click 'Run!'. "
            "You can also iterate on follow-ups until the chart is right."
        )

        st.markdown("#### How it works")
        st.markdown(
            "**1. Load your data.** Upload a CSV, TXT, XLSX, or JSON file from the sidebar, "
            "or pick one of the built-in example datasets to try the tool without uploading anything."
        )
        st.markdown(
            "**2. Describe the chart in plain English.** Include the chart type and column names "
            "when you can. The agent reads your data's shape and picks an appropriate chart type "
            "even if you don't specify one."
        )
        st.markdown(
            "**3. Run and iterate.** Review the plan, generated code, and resulting chart. "
            "Refine with follow-up prompts like \"log-scale the y-axis\" or \"make the bars red\"."
        )

        st.markdown("#### Built-in examples (select 'Example' button in the sidebar)")
        st.markdown(
            "- 🌸 **Iris** — 150 rows of flower measurements across 3 species. Classic multivariate dataset.\n"
            "- 📈 **Sales timeseries** — 1,095 days of revenue by region with trend, seasonality, and noise.\n"
            "- 📊 **Survey** — 500 employees with age, income, satisfaction, and department fields."
        )
        st.caption("→ Open the sidebar, choose **Example**, pick one, and click **Load example**.")


def render_ready(container, df):
    with container.container():
        st.markdown("### 📊 Ready to visualize")
        st.markdown(
            f"Loaded **{len(df):,} rows × {len(df.columns)} columns**. "
            "Describe what you want to plot in the sidebar."
        )
        # suggestions = suggest_prompts(df)
        # if suggestions:
        #     st.markdown("**Suggested prompts for this dataset:**")
        #     for s in suggestions:
        #         st.markdown(f"- `{s}`")
        #     st.caption("Copy one into the sidebar, tweak it, or write your own 😊.")


# Decide what to show in the chart area BEFORE running the pipeline
if st.session_state.df is None:
    render_welcome(chart_area)
elif run_button and not over_limit and user_prompt.strip():
    with chart_area.container():
        st.info("⏳ Generating visualization — pipeline details below.")
elif st.session_state.last_fig is not None:
    render_chart_panel(
        chart_area,
        st.session_state.last_fig,
        st.session_state.last_plan,
        st.session_state.last_code,
        st.session_state.last_prompt,
        turn_id=f"prev_{len(st.session_state.history)}",
    )
else:
    render_ready(chart_area, st.session_state.df)

# Data preview — below chart area. Expanded when no chart yet, collapsed once we have one.
if st.session_state.df is not None:
    
    # Expandable instructions section
    with st.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        **Workflow**
        1. **Load data** — upload a file or pick an example.
        2. **Describe** the chart — mention chart type and column names if you know them. Be as specific as possible.
        3. **Run** — the agent plans, writes code, and executes it (with up to 3 debug retries).
        4. **Refine** — tick 🔁 Refine and send a follow-up like "log-scale the y-axis".

        **Example prompts**
        - `line chart of revenue over time`
        - `scatter plot of height vs weight`
        - `box plot of exam_score by study_hours`
        - `distribution of income`
        - `grouped bar chart of counts by department and satisfaction`

        **Notes**
        - Generated code is AST-checked.
        - Downloads (PNG, SVG, Python code) appear with each successful run.
        - 👍 / 👎 feedback is logged for evaluation.
        """)
        
    # Expandable dataset preview section
    with st.expander("📋 Data preview", expanded=(st.session_state.last_fig is None)):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(st.session_state.df):,}")
        c2.metric("Columns", len(st.session_state.df.columns))
        c3.metric("Memory", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        tab_data, tab_summary = st.tabs(["Data", "LLM summary"])
        with tab_data:
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
        with tab_summary:
            st.code(st.session_state.df_summary)
    

# Run pipeline — status blocks render HERE and auto-collapse on success
if run_button and st.session_state.df is not None and user_prompt.strip():
    # prev_plan = st.session_state.last_plan if refine else None
    # prev_code = st.session_state.last_code if refine else None

    # fig, code, plan = run_pipeline(
    #     st.session_state.df,
    #     st.session_state.df_summary,
    #     user_prompt,
    #     previous_plan=prev_plan,
    #     previous_code=prev_code,
    # )

    prev_plan = st.session_state.last_plan if refine else None
    prev_code = st.session_state.last_code if refine else None

    run_id = uuid.uuid4().hex[:10]
    t0 = time.time()
    fig, code, plan = run_pipeline(
        st.session_state.df, st.session_state.df_summary, user_prompt,
        previous_plan=prev_plan, previous_code=prev_code,
    )
    elapsed = time.time() - t0

    # exec attempts = failed executions + the successful one (if any)
    attempts = sum(1 for e in st.session_state.get("debug_log", [])
                   if e["label"].startswith("Exec error")) + (1 if fig is not None else 0)
    log_run(run_id, user_prompt, plan, code, fig is not None, elapsed, st.session_state.df, attempts, refine = refine, source = source)
    st.session_state.used_suggestion = False
    st.session_state.last_run_id = run_id
    # New code^

    if fig is not None:
        st.session_state.last_fig = fig
        st.session_state.last_plan = plan
        st.session_state.last_code = code
        st.session_state.last_prompt = user_prompt
        st.session_state.history.append({
            "prompt": user_prompt,
            "plan": plan,
            "code": code,
        })
        # Pin the new chart to the top area
        render_chart_panel(
            chart_area, fig, plan, code, user_prompt,
            turn_id=f"new_{len(st.session_state.history)}",
        )
        st.toast("Visualization complete", icon="✅")
    else:
        if not st.session_state.get("refused"):
            with chart_area.container():
                st.error("❌ Could not generate a chart. Expand the pipeline details above to see what happened, then tweak your prompt and try again.")

# Generated code viewer — collapsed, below everything
# if st.session_state.last_code:
#     with st.expander("💻 Generated code", expanded=False):
#         st.code(st.session_state.last_code, language="python")

# History panel
if st.session_state.history:
    with st.expander(f"🕘 Conversation history ({len(st.session_state.history)} turns)", expanded=False):
        for i, turn in enumerate(reversed(st.session_state.history), 1):
            turn_num = len(st.session_state.history) - i + 1
            st.markdown(f"**Turn {turn_num}:** {turn['prompt']}")
            st.caption(f"chart: `{turn['plan'].get('chart_type')}` · cols: `{turn['plan'].get('relevant_columns')}`")
            st.divider()

# Debug panel
if st.session_state.get("debug_log"):
    with st.expander("🪲 Debug log (raw LLM output)", expanded=False):
        for entry in st.session_state.debug_log:
            icon = "🔴" if entry["level"] == "error" else "🟢"
            st.markdown(f"**{icon} {entry['label']}**")
            st.code(entry["content"])
            st.divider()