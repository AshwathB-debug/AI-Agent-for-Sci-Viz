# ==============================================
# BLOCK 2: LLM AGENTS (Thinker, Coder, Debugger)
# ==============================================


from dotenv import load_dotenv
from openai import OpenAI
# from huggingface_hub import InferenceClient
import os, json, re, time
from config import VALID_CHART_TYPES, Union, Dict, Any, Optional, Tuple, List
# import streamlit as st


load_dotenv()
client = OpenAI(
    base_url = "https://router.huggingface.co/v1",
    api_key = os.environ["APIKEY"],
    timeout = 60,
    max_retries = 3, # retries transient errors like 429/5xx with backoff
)



class LLMError(RuntimeError):
    pass


class Qwen:
    
    def __init__(self):
        self.usage_log = []
        # st.session_state.fail_stage = ""
        
    
    """Thinker / Coder / Debugger wrappers around a single loaded Qwen model."""
    def run(self, messages, max_new_tokens = 1024, temperature: float = 0.0) -> Optional[str]:
        # maxRetries = 3
        # backoffFactor = 2
        
        """Unified generate; do_sample flag derived from temperature to avoid contradictions."""
        
        # for attempt in range(maxRetries):        
        try:
            
            response = client.chat.completions.create(
                model = "Qwen/Qwen3-32B:cheapest",
                messages = messages,
                max_tokens = max_new_tokens,
                temperature = temperature,
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}},
            )
            
            # Guard against empty choices or missing message content to prevent IndexError/AttributeError
            if not response.choices or not response.choices[0].message:
                raise LLMError("Received empty or malformed response structure from API.")
            
            # Guard against missing or None usage data from API router
            if hasattr(response, "usage") and response.usage:
                self.usage_log.append({
                    "prompt": getattr(response.usage, "prompt_tokens", 0),
                    "completion": getattr(response.usage, "completion_tokens", 0)
                })
            return self._clean(response.choices[0].message.content) or ""
        
        except Exception as e:
            # if attempt == maxRetries - 1:
            #     # Final attempt failed, return None so the UI handles it gracefully
            #     return None
            # time.sleep(backoffFactor ** attempt)  # Wait 2s, 4s before retrying
            
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"{type(e).__name__}: {e}") from e    
        # return None
        

    @staticmethod
    def _clean(text: str) -> str:
        
        if not text:
            return text
        cleaned = text.strip()
        cleaned = re.sub(r"```(?:json|python)?", "", cleaned, flags = re.IGNORECASE)
        cleaned = re.sub(r"<think>.*?(</think>|$)", "", cleaned, flags = re.DOTALL)
        cleaned = cleaned.replace("```", "").strip()
        return cleaned


    def thinker(self, df_summary: str, user_prompt: str, previous_plan: Optional[Dict] = None, previous_code: Optional[str] = None) -> str:
        
        refinement_block = ""
        
        if previous_plan and previous_code:
            refinement_block = f"""
            REFINEMENT MODE: The user is iterating on the previous visualization.
            Previous plan: {json.dumps(previous_plan)}
            Previous code produced a working chart. The user now wants adjustments.
            Reuse the previous chart_type and columns UNLESS the new request explicitly asks for different ones.
            """

        system_msg = f"""
        You are a Data Visualization Architect. Output ONLY valid JSON — no prose, no markdown fences.

        Before choosing the chart, fill the "reasoning" field by walking through:
        <reasoning_process>
        (a) What is the x-axis column? dtype, unique count, and is it ordered/temporal/categorical/continuous?
        (b) What is the y-axis column? dtype, continuous or categorical?
        (c) How many rows per x-value (duplicates? one-per-x?)?
        (d) Which rule below matches? State it explicitly.
        </reasoning_process>
        

        Apply in order — first match wins:
        
        <chart_type_rules>
        A. TIME SERIES → "line"
        When x is datetime/date/year or a monotonically ordered index.

        B. LOW-CARDINALITY X WITH DUPLICATES → "box" or "violin"
        When x has < 20 unique values AND > 5 rows per value AND y is continuous.
        Prefer "box" for comparison, "violin" when distribution shape matters.
        CRITICAL: NEVER use "line" here — it produces a meaningless zig-zag through duplicate x-values.

        C. CATEGORICAL COUNTS / FREQUENCIES → "bar"
        When user asks for counts, frequencies, or "how many". Use value_counts() or pd.crosstab() in coder_instructions.

        D. CATEGORICAL MEANS → "bar"
        When x is categorical (<20 unique), y is numeric continuous, user wants averages. Use df.groupby('x')['y'].mean() in coder_instructions.

        E. TWO CONTINUOUS NUMERIC → "scatter"
        Both x and y are continuous with many unique values. Suggest alpha=0.3 if rows > 1000.

        F. SINGLE-COLUMN DISTRIBUTION → "histogram"
        User wants to see the shape of one numeric column.

        G. MANY NUMERIC COLUMNS, CORRELATION → "heatmap"

        H. PART-OF-WHOLE → "pie"
        ONLY when x has ≤7 categories. With >7, use "bar" instead — pies become unreadable.
        </chart_type_rules>
        
        <hard_constraints>
        - USER INTENT PRECEDENCE: IF the user explicitly names a chart type (e.g. "bar chart", "line chart", "scatter plot"), you MUST use that chart_type. Do NOT override it with a different chart type unless it is mathematically impossible.
        - chart_type MUST be one of: {VALID_CHART_TYPES}
        - relevant_columns MUST be exact column names from the dataset summary.
        - Text categorical columns (e.g. 'Low','Medium','High','Yes','No') → NEVER compute mean. Use counts/crosstab.
        - When x has duplicates AND chart is line or bar, coder_instructions MUST specify aggregation (groupby + mean/sum).
        </hard_constraints>
        
        
        <examples>
        --- Example 1: time series ---
        Dataset: 1095 rows × 3 cols. Columns: 'date' (datetime64, 0 nulls), 'revenue' (float, 0 nulls), 'region' (object, 4 unique).
        Request: "line chart of revenue over time"
        {{
        "reasoning": "x=date is datetime → Rule A (time series) → line. Multiple rows per date (one per region) so aggregate by summing across regions.",
        "task_summary": "Total daily revenue across all regions",
        "relevant_columns": ["date", "revenue"],
        "human_readable_labels": {{"title": "Daily Revenue Over Time", "x_label": "Date", "y_label": "Revenue ($)"}},
        "chart_type": "line",
        "coder_instructions": "Aggregate: df.groupby('date')['revenue'].sum().reset_index(), then sort by date. Rotate x-ticks 45°. Format y-axis with K/M FuncFormatter."
        }}

        --- Example 2: low-cardinality x with duplicates ---
        Dataset: 6607 rows × 20 cols. Key columns: 'Sleep_Hours' (int64, 7 unique: 4-10), 'Exam_Score' (int64, ~50 unique, range 55-101).
        Request: "plot exam score vs sleep hours"
        {{
        "reasoning": "x=Sleep_Hours has 7 unique integer values but 6607 rows → ~900 rows per value. y=Exam_Score is continuous. Rule B → box (distribution per level). Line would zig-zag meaninglessly.",
        "task_summary": "Distribution of exam scores at each sleep-hours level",
        "relevant_columns": ["Sleep_Hours", "Exam_Score"],
        "human_readable_labels": {{"title": "Exam Score Distribution by Sleep Hours", "x_label": "Sleep Hours", "y_label": "Exam Score"}},
        "chart_type": "box",
        "coder_instructions": "Build a box plot of Exam_Score grouped by Sleep_Hours using df.boxplot(column='Exam_Score', by='Sleep_Hours'). Do not aggregate — the boxplot computes its own summary statistics."
        }}

        --- Example 3: categorical counts (crosstab) ---
        Dataset: 500 rows × 4 cols. 'department' (object, 4 unique: Eng/Sales/HR/Ops), 'satisfaction' (object, 3 unique: Low/Medium/High).
        Request: "bar chart of satisfaction by department"
        {{
        "reasoning": "Both columns are text categorical → cannot compute mean (violates hard constraint). User wants counts per (dept, satisfaction) pair → Rule C → grouped bars via crosstab.",
        "task_summary": "Satisfaction counts broken down by department",
        "relevant_columns": ["department", "satisfaction"],
        "human_readable_labels": {{"title": "Satisfaction Counts by Department", "x_label": "Department", "y_label": "Count"}},
        "chart_type": "bar",
        "coder_instructions": "ct = pd.crosstab(df['department'], df['satisfaction']); then ct.plot(kind='bar', ax=plt.gca()). Do NOT loop with plt.bar."
        }}

        --- Example 4: two continuous numeric ---
        Dataset: 200 rows × 5 cols. 'height_cm' (float, ~180 unique, 150-195), 'weight_kg' (float, ~175 unique, 45-110).
        Request: "relationship between height and weight"
        {{
        "reasoning": "Both x and y continuous with many unique values → Rule E → scatter. 200 rows is small, no alpha needed.",
        "task_summary": "Relationship between height and weight",
        "relevant_columns": ["height_cm", "weight_kg"],
        "human_readable_labels": {{"title": "Weight vs Height", "x_label": "Height (cm)", "y_label": "Weight (kg)"}},
        "chart_type": "scatter",
        "coder_instructions": "Scatter plot height_cm (x) vs weight_kg (y). Use plt.margins(0.05)"
        }}

        --- Example 5: single-column distribution ---
        Dataset: 1000 rows × 3 cols. 'income' (float, range 20000-250000), 'age' (int), 'city' (object).
        Request: "what does the income distribution look like"
        {{
        "reasoning": "One continuous numeric column, wide range hints at skew → Rule F → histogram.",
        "task_summary": "Shape of the income distribution",
        "relevant_columns": ["income"],
        "human_readable_labels": {{"title": "Income Distribution", "x_label": "Income ($)", "y_label": "Frequency"}},
        "chart_type": "histogram",
        "coder_instructions": "plt.hist(df['income'].dropna(), bins=30). Format x-axis with K/M FuncFormatter."
        }}
        </examples>
        {refinement_block}
        
        <assistant_turn>
        Dataset Summary:
        {df_summary}

        User Request: {user_prompt}

        Output ONLY this JSON structure (reasoning first, then the plan):
        {{
        "reasoning": "step-by-step analysis following the reasoning process above",
        "task_summary": "Brief explanation of the goal",
        "relevant_columns": ["col1", "col2"],
        "human_readable_labels": {{
            "title": "A descriptive title",
            "x_label": "Clean X-axis label",
            "y_label": "Clean Y-axis label (include units like $ or % if applicable)"
        }},
        "chart_type": "one of the allowed types",
        "coder_instructions": "Technical instructions for the coder"
        }}
        </assistant_turn>
        """
        
        return self.run([{"role": "user", "content": system_msg}], temperature=0.0, max_new_tokens=1400)


    def coder(self, df_summary: str, plan_json: Dict, previous_code: Optional[str] = None) -> str:
        
        cols = plan_json.get('relevant_columns', [])
        chart_type = plan_json.get('chart_type', 'plot')
        instructions = plan_json.get('coder_instructions', 'Visualize the data.')
        labels = plan_json.get('human_readable_labels', {})
        title = labels.get('title', 'Data Visualization')
        x_label = labels.get('x_label', cols[0] if cols else 'X')
        y_label = labels.get('y_label', cols[1] if len(cols) > 1 else 'Y')

        refinement_block = ""
        if previous_code:
            refinement_block = f"""
            REFINEMENT: Start from this previously-working code and modify it to match the new plan.
            --- PREVIOUS CODE ---
            {previous_code}
            --- END PREVIOUS CODE ---
            """

        system_msg = f"""
        
        You are a Python Visualization Expert. Write a code snippet that follows ALL rules below.

        <critical_rules>
        1. NO DATA LOADING. `df` is already in memory. NEVER call pd.read_csv.
        2. NO IMPORTS. `plt`, `pd`, `np` are already imported.
        3. COLUMN DISCIPLINE: use ONLY columns listed in relevant_columns. If only 2 are requested, DO NOT introduce a 3rd for color or size.
        4. HUMAN LABELS: you MUST use the provided strings:
            plt.title({title!r})
            plt.xlabel({x_label!r})
            plt.ylabel({y_label!r})
        5. SCALING: do NOT call set_xlim/set_ylim. Use plt.margins(0.05) for padding.
            Only set limits if the user explicitly asks for a range or log scale.
        6.  LARGE NUMBERS (>10,000): format y-axis with matplotlib.ticker.FuncFormatter into K/M/B. Do NOT leave scientific notation.
        7.  DATETIME BAR CHARTS: compute the minimum day-difference between sorted dates and set width=min_diff_days*0.8 in plt.bar() so bars don't vanish.
        8.  TIME LABELS: plt.xticks(rotation=45) whenever x-axis is a date.
        9. AGGREGATION: if the dataset has MULTIPLE ROWS PER X-VALUE and the chart is line or bar, you MUST aggregate first. Use df.groupby('X')['Y'].mean().reset_index() for averages (default) or .sum() when the user asks for totals. This applies to ANY duplicate x-values, not just dates. Skip this rule for scatter, box, violin, and histogram — those handle duplicates natively.
        10. LINE CHARTS: sort by the x-axis first with df.sort_values().
        11. GROUPED BARS: for crosstabs/multi-category bars, use df.plot(kind='bar', ax=plt.gca()) — NOT a for-loop with plt.bar.
        12. VISIBILITY: use alpha = 0.5 for scatter plots with > 1000 rows.
        13. OUTPUT: raw Python code only. No markdown fences, no commentary, no explanations.
        14. LABEL READIBILITY: If labels are categorical, make labels vertical.
        15. LEGENDS: When adding a legend, if there are many items, use multi-column layouts (e.g., `ncol = 2` or `ncol = 3`) or anchor it cleanly outside using `plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')`. If a categorical column has excessive unique values (>10), avoid grouping by it to prevent layout distortion.
        16. FILTERING: only filter when the user gives explicit values or conditions
            (e.g. "age 41", "only India vs New Zealand"). If they just mention entities,
            include rows where the entity appears in EITHER column, e.g.
            df[(df['Team_A'].isin(teams)) | (df['Team_B'].isin(teams))].
        17. TYPE SAFETY FOR AGGREGATION: NEVER call `.mean()`, `.sum()`, or numeric aggregations on text/string object columns. If you are grouping by a text category to see distributions or frequencies, use `.size().unstack()` or `.value_counts()` instead of numeric mean operations.
        18. DATETIME FILTERING: If a date column has been parsed as datetime64, NEVER use `.str.startswith()` or string methods on it. Instead, use pandas datetime accessors or filtering (e.g. `(df['Date'].dt.year == 2026) & (df['Date'].dt.month == 2)`).
        {refinement_block}
        </critical_rules>
        
        <assistant_turn>
        Dataset Summary:
        {df_summary}

        Plan:
        Create a {chart_type} using columns {cols}.
        Instructions: {instructions}

        Write the code.
        </assistant_turn>
        """
        return self.run([{"role": "user", "content": system_msg}], max_new_tokens = 2000, temperature=0.0)


    def debugger(self, df_summary: str, plan_json: Dict, failed_code: str, error_msg: str) -> str:
        
        cols = plan_json.get('relevant_columns', [])
        chart_type = plan_json.get('chart_type', 'plot')

        # Truncate huge tracebacks so we don't blow context
        if len(error_msg) > 2000:
            error_msg = error_msg[:1000] + "\n...[truncated]...\n" + error_msg[-800:]

        system_msg = """
        You are a Python Debugging Expert. Fix the matplotlib/pandas script that crashed.

        <rules>
        1. NO APOLOGIES, NO EXPLANATIONS, NO MARKDOWN.
        2. Return ONLY the corrected Python code.
        3. Do NOT use pd.read_csv. `df` is already loaded.
        4. Do NOT add imports. `plt`, `pd`, `np` are available.
        5. Preserve the original goal; fix the exact line(s) that failed.
        6. For KeyError/TypeError on dates/strings: coerce or drop problematic rows safely before plotting.
        </rules>
        """
        
        user_msg = f"""
        Dataset Summary:
        {df_summary}

        Original Goal: create a {chart_type} using columns {cols}.

        --- FAILED CODE ---
        {failed_code}
        --- END FAILED CODE ---

        --- ERROR ---
        {error_msg}
        --- END ERROR ---

        Output the corrected full script."""

        # Slightly higher temp so it tries different approaches on repeat failures
        return self.run(
            [{"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}],
            max_new_tokens = 2000,
            temperature = 0.2,
        )