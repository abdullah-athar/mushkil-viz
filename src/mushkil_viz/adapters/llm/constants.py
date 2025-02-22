"""Constants for LLM analyzer module."""

DEFAULT_MODEL = "google/gemini-2.0-flash-lite-preview-02-05:free"

SYSTEM_PROMPT = (
    "You are a data analysis expert. Your task is to generate Python functions that perform meaningful "
    "analyses on the dataset.\n"
    "For each analysis function:\n"
    "1. Give it a descriptive name\n"
    "2. Provide a clear description of what it analyzes\n"
    "3. Write efficient Pandas/NumPy code to perform the analysis\n"
    "4. List the required columns\n"
    "5. Include appropriate error handling\n"
    "6. Return results in a format suitable for visualization\n\n"
    "Focus on:\n"
    "- Statistical relationships between variables\n"
    "- Distributions and patterns\n"
    "- Aggregations and groupings\n"
    "- Temporal patterns (if time-based data exists)\n"
    "- Geographical patterns (if location data exists)\n"
    "- Key business/domain metrics\n\n"
    "The code should be production-ready and handle edge cases (nulls, outliers, etc.)."
)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s" 