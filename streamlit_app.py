import streamlit as st
import pandas as pd
import numpy as np
import requests

# ---------- Finance math functions ----------
def _trim_zeros(s: str) -> str:
    s = s.rstrip('0').rstrip('.')
    return s if s else "0"

def toWords(num: float, decimals: int = 2) -> str:
    if num is None:
        return "0"
    neg = num < 0
    n = abs(float(num))
    if n >= 1e7:
        val, unit = n / 1e7, "Cr."
    elif n >= 1e5:
        val, unit = n / 1e5, "Lakh"
    elif n >= 1e3:
        val, unit = n / 1e3, "k"
    else:
        text = str(int(round(n)))
        return f"-{text}" if neg and text != "0" else text
    text = _trim_zeros(f"{val:.{decimals}f}")
    return f"-{text} {unit}" if neg else f"{text} {unit}"

def inflation_adjust(value: float, inflation_rate: float, years: int) -> float:
    return value / ((1 + inflation_rate) ** years)

def inflate_future(value: float, inflation_rate: float, years: int) -> float:
    return value * ((1 + inflation_rate) ** years)

def fv_growing_annuity_monthly(monthly_pmt: float, g: float, r: float, n_years: int) -> float:
    future_value = 0
    monthly_rate = r / 12.0
    for year in range(n_years):
        current_year_monthly_pmt = monthly_pmt * ((1 + g) ** year)
        if monthly_rate == 0:
            fv_this_year = current_year_monthly_pmt * 12
        else:
            fv_this_year = current_year_monthly_pmt * (((1 + monthly_rate) ** 12 - 1) / monthly_rate)
        future_value += fv_this_year * ((1 + monthly_rate) ** (12 * (n_years - (year + 1))))
    return future_value

def nps_calculator(monthly_investment: float, annual_increase: float, years: int, expected_return: float) -> float:
    return fv_growing_annuity_monthly(monthly_investment, annual_increase, expected_return, years)

def nps_withdrawal_pension(
    monthly_investment: float,
    annual_increase: float,
    years: int,
    expected_return: float,
    withdrawal_percent: float,
    annuity_return: float,
    life_expectancy: int = 20 * 12,
    inflation_rate: float = 0.06,
    target_pension_today: float | None = None,
    words_decimals: int = 2
):
    corpus = nps_calculator(monthly_investment, annual_increase, years, expected_return)
    w = max(0.0, min(1.0, withdrawal_percent))
    lumpsum = corpus * w
    annuity_corpus = corpus - lumpsum

    monthly_rate = annuity_return / 12.0
    months = max(1, int(life_expectancy))
    if monthly_rate > 0:
        pension = annuity_corpus * monthly_rate / (1 - (1 + monthly_rate) ** (-months))
        pension_factor = monthly_rate / (1 - (1 + monthly_rate) ** (-months))
    else:
        pension = annuity_corpus / months
        pension_factor = 1.0 / months

    adj_corpus = inflation_adjust(corpus, inflation_rate, years)
    adj_lumpsum = inflation_adjust(lumpsum, inflation_rate, years)
    adj_annuity_corpus = inflation_adjust(annuity_corpus, inflation_rate, years)
    adj_pension = inflation_adjust(pension, inflation_rate, years)

    result = {
        "Total Corpus": (corpus, toWords(corpus, words_decimals), adj_corpus, toWords(adj_corpus, words_decimals)),
        "Lumpsum": (lumpsum, toWords(lumpsum, words_decimals), adj_lumpsum, toWords(adj_lumpsum, words_decimals)),
        "Annuity Corpus": (annuity_corpus, toWords(annuity_corpus, words_decimals), adj_annuity_corpus, toWords(adj_annuity_corpus, words_decimals)),
        "Monthly Pension": (pension, toWords(pension, words_decimals), adj_pension, toWords(adj_pension, words_decimals)),
    }

    if target_pension_today is not None:
        target_future = inflate_future(target_pension_today, inflation_rate, years)
        gap = max(0.0, target_future - pension)
        A = fv_growing_annuity_monthly(1.0, annual_increase, expected_return, years)
        annuity_per_invest = (1.0 - w) * A
        pension_per_invest = annuity_per_invest * pension_factor
        extra_annual = (gap / pension_per_invest) * 12 if pension_per_invest > 0 and np.isfinite(pension_per_invest) else float('inf')
        extra_monthly = gap / pension_per_invest if pension_per_invest > 0 and np.isfinite(pension_per_invest) else float('inf')
        result["Pension Gap Analysis"] = {
            "Target Monthly Pension (Today's ₹)": target_pension_today,
            "Target Monthly Pension (Future ₹)": target_future,
            "Current Monthly Pension (Future ₹)": pension,
            "Pension Shortfall (Future ₹)": gap,
            "Extra Annual Investment Needed": max(0.0, extra_annual) if np.isfinite(extra_annual) else float('inf'),
            "Extra Monthly Investment Needed": max(0.0, extra_monthly) if np.isfinite(extra_monthly) else float('inf'),
            "Words": {
                "Target Monthly Pension (Today's ₹)": toWords(target_pension_today, words_decimals),
                "Target Monthly Pension (Future ₹)": toWords(target_future, words_decimals),
                "Current Monthly Pension (Future ₹)": toWords(pension, words_decimals),
                "Pension Shortfall (Future ₹)": toWords(gap, words_decimals),
                "Extra Annual Investment Needed": toWords(extra_annual, words_decimals) if np.isfinite(extra_annual) else "∞",
                "Extra Monthly Investment Needed": toWords(extra_monthly, words_decimals) if np.isfinite(extra_monthly) else "∞",
            }
        }
    return result

def epf_calculator(monthly_investment: float, annual_interest_rate: float, years: int, annual_increase: float = 0) -> float:
    monthly_rate = annual_interest_rate / 12
    future_value = 0
    for year in range(years):
        current_year_monthly_investment = monthly_investment * ((1 + annual_increase) ** year)
        if monthly_rate == 0:
            fv_this_year = current_year_monthly_investment * 12
        else:
            fv_this_year = current_year_monthly_investment * (((1 + monthly_rate) ** 12 - 1) / monthly_rate)
        future_value += fv_this_year * ((1 + monthly_rate) ** (12 * (years - (year + 1))))
    return future_value

def ppf_calculator(annual_investment: float, annual_interest_rate: float, years: int) -> float:
    max_annual_investment = 150000
    annual_investment = min(annual_investment, max_annual_investment)
    future_value = 0
    for year in range(years):
        fv_this_year = annual_investment * ((1 + annual_interest_rate) ** (years - (year + 1)))
        future_value += fv_this_year
    return future_value

# ---------- Streamlit App ----------

st.set_page_config(
    page_title="Comprehensive Retirement & Investment Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💰 Comprehensive Retirement & Investment Calculator")

with st.expander("ℹ️ How to use this tool", expanded=False):
    st.markdown("""
    1. Adjust the input parameters in the sidebar.
    2. Explore the results below, including metrics, tables, and graphs.
    3. Review the AI recommendations to optimize your financial plan.
    """)

st.sidebar.header("🔧 Master Inputs")
st.sidebar.write("These inputs apply to all calculations and represent your overall retirement planning horizon and economic assumptions.")
ms_years = st.sidebar.number_input("Years to Retirement", min_value=1, value=37, help="Years until you plan to retire.")
ms_annual_increase = st.sidebar.number_input("Annual Increase in Investment (%)", min_value=0.0, value=10.0, help="Yearly % increase in investments.") / 100.0
ms_inflation_rate = st.sidebar.number_input("Inflation Rate (%)", min_value=0.0, value=7.0, help="Avg annual inflation rate.") / 100.0

st.sidebar.header("🛡️ NPS Inputs")
st.sidebar.write("Specify the details for your NPS investments.")
ms_nps_annual_investment = st.sidebar.number_input("Annual Investment (₹)", min_value=0, value=96589)
ms_nps_expected_return = st.sidebar.number_input("Expected Return (%)", min_value=0.0, value=12.0) / 100.0
ms_nps_lumpsum_withdrawal_percent = st.sidebar.number_input("Lumpsum Withdrawal (%)", min_value=0.0, max_value=100.0, value=0.0) / 100.0
ms_nps_annuity_return = st.sidebar.number_input("Annuity Return (%)", min_value=0.0, value=7.0) / 100.0
ms_nps_life_expectancy = st.sidebar.number_input("Life Expectancy (Years)", min_value=1, value=20)
ms_targeted_pension_today = st.sidebar.number_input("Target Monthly Pension (Today's ₹)", min_value=0, value=100000)

st.sidebar.header("🛡️ EPF Inputs")
st.sidebar.write("Specify the details for your EPF contributions.")
ms_epf_monthly_investment = st.sidebar.number_input("Monthly Investment (₹)", min_value=0, value=15000)
ms_epf_annual_interest_rate = st.sidebar.number_input("Annual Interest Rate (%)", min_value=0.0, value=8.25) / 100.0

st.sidebar.header("🛡️ PPF Inputs")
st.sidebar.write("Specify the details for your PPF investments.")
ms_ppf_annual_investment = st.sidebar.number_input("Annual Investment (₹)", min_value=0, value=150000)
ms_ppf_annual_interest_rate = st.sidebar.number_input("Annual Interest Rate (%)", min_value=0.0, value=7.1) / 100.0

# ---------- Calculations ----------
nps_result = nps_withdrawal_pension(
    monthly_investment = ms_nps_annual_investment / 12,
    annual_increase = ms_annual_increase,
    years = ms_years,
    expected_return = ms_nps_expected_return,
    withdrawal_percent = ms_nps_lumpsum_withdrawal_percent,
    annuity_return = ms_nps_annuity_return,
    life_expectancy = ms_nps_life_expectancy * 12,
    inflation_rate = ms_inflation_rate,
    target_pension_today = ms_targeted_pension_today,
    words_decimals=2
)

epf_future_value = epf_calculator(
    monthly_investment = ms_epf_monthly_investment,
    annual_interest_rate = ms_epf_annual_interest_rate,
    years = ms_years,
    annual_increase = ms_annual_increase
)
epf_inflation_adjusted_value = inflation_adjust(epf_future_value, ms_inflation_rate, ms_years)

ppf_future_value = ppf_calculator(
    annual_investment = ms_ppf_annual_investment,
    annual_interest_rate = ms_ppf_annual_interest_rate,
    years = ms_years
)
ppf_inflation_adjusted_value = inflation_adjust(ppf_future_value, ms_inflation_rate, ms_years)

# ---------- Display Metrics ----------
st.header("📊 Key Retirement Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="NPS Corpus (Future ₹)",
        value=f"{nps_result['Total Corpus'][1]}",
        help="Projected NPS corpus at retirement"
    )
with col2:
    st.metric(
        label="NPS Monthly Pension (Future ₹)",
        value=f"{nps_result['Monthly Pension'][1]}",
        help="Projected monthly pension from NPS"
    )
with col3:
    st.metric(
        label="EPF Corpus (Future ₹)",
        value=f"₹{toWords(epf_future_value)}",
        help="Projected EPF corpus at retirement"
    )
with col4:
    st.metric(
        label="PPF Corpus (Future ₹)",
        value=f"{toWords(ppf_future_value)}",
        help="Projected PPF corpus at retirement"
    )

# ---------- Color coded pension gap ----------
if "Pension Gap Analysis" in nps_result:
    gap = nps_result["Pension Gap Analysis"]["Pension Shortfall (Future ₹)"]
    extra_monthly = nps_result["Pension Gap Analysis"]["Extra Monthly Investment Needed"]
    if gap > 0:
        st.error(f"**⚠️ NPS Pension Shortfall:** Your projected pension is ₹{gap:,.0f} short of your target.\n\nConsider investing ₹{extra_monthly:,.0f} more per month.")
    else:
        st.info(f"**✅ NPS Target Met:** Your projected pension meets your goal.\n\nYou may consider increasing your target or retiring earlier!")

# ---------- Graphs ----------
st.header("📈 Investment Growth Over Time")

years_range = np.arange(1, ms_years + 1)

# NPS: Future value for each year
nps_corpus_each_year = [
    nps_calculator(
        monthly_investment=ms_nps_annual_investment / 12,
        annual_increase=ms_annual_increase,
        years=year,
        expected_return=ms_nps_expected_return
    )
    for year in years_range
]

# EPF: Future value for each year
epf_corpus_each_year = [
    epf_calculator(
        monthly_investment=ms_epf_monthly_investment,
        annual_interest_rate=ms_epf_annual_interest_rate,
        years=year,
        annual_increase=ms_annual_increase
    )
    for year in years_range
]

# PPF: Future value for each year
ppf_corpus_each_year = [
    ppf_calculator(
        annual_investment=ms_ppf_annual_investment,
        annual_interest_rate=ms_ppf_annual_interest_rate,
        years=year
    )
    for year in years_range
]

df_growth = pd.DataFrame({
    "Year": years_range,
    "NPS Corpus": nps_corpus_each_year,
    "EPF Corpus": epf_corpus_each_year,
    "PPF Corpus": ppf_corpus_each_year,
    "Total Corpus": np.array(nps_corpus_each_year) + np.array(epf_corpus_each_year) + np.array(ppf_corpus_each_year)
})

st.line_chart(df_growth.set_index("Year"), use_container_width=True)

# ---------- Calculation Tables ----------
st.header("🧮 Detailed Calculation Tables")
with st.expander("NPS Calculation Details", expanded=True):
    nps_summary_data = {
        "Metric": [k for k in nps_result if k != "Pension Gap Analysis"],
        "Future Value (₹)": [v[0] for k,v in nps_result.items() if k != "Pension Gap Analysis"],
        "Future Value": [v[1] for k,v in nps_result.items() if k != "Pension Gap Analysis"],
        "Inflation Adjusted Value (Today's ₹)": [v[2] for k,v in nps_result.items() if k != "Pension Gap Analysis"],
        "Inflation Adjusted Value": [v[3] for k,v in nps_result.items() if k != "Pension Gap Analysis"]
    }
    nps_summary_df = pd.DataFrame(nps_summary_data)
    st.dataframe(nps_summary_df, use_container_width=True)

    if "Pension Gap Analysis" in nps_result:
        st.markdown("**Pension Gap Analysis**")
        gap_data = nps_result["Pension Gap Analysis"]["Words"]
        gap_summary_data = {
            "Metric": [k for k in gap_data],
            "Value": [v for k,v in gap_data.items()]
        }
        gap_summary_df = pd.DataFrame(gap_summary_data)
        st.dataframe(gap_summary_df, use_container_width=True)

with st.expander("EPF Calculation Details", expanded=False):
    epf_summary_data = {
        "Metric": ["Future Value of EPF Corpus", "Inflation Adjusted EPF Value"],
        "Value (₹)": [epf_future_value, epf_inflation_adjusted_value],
        "Value": [toWords(epf_future_value), toWords(epf_inflation_adjusted_value)]
    }
    epf_summary_df = pd.DataFrame(epf_summary_data)
    st.dataframe(epf_summary_df, use_container_width=True)

with st.expander("PPF Calculation Details", expanded=False):
    ppf_summary_data = {
        "Metric": ["Future Value of PPF", "Inflation Adjusted PPF Value"],
        "Value (₹)": [ppf_future_value, ppf_inflation_adjusted_value],
        "Value": [toWords(ppf_future_value), toWords(ppf_inflation_adjusted_value)]
    }
    ppf_summary_df = pd.DataFrame(ppf_summary_data)
    st.dataframe(ppf_summary_df, use_container_width=True)

# ---------- Investment Summary Table ----------
st.header("💡 Investment Summary")
total_nps_invested = sum(nps_corpus_each_year)
total_epf_invested = sum(epf_corpus_each_year)
total_ppf_invested = sum(ppf_corpus_each_year)
total_invested = total_nps_invested + total_epf_invested + total_ppf_invested
current_nps_value = nps_result["Total Corpus"][0]
current_epf_value = epf_future_value
current_ppf_value = ppf_future_value
total_current_value = current_nps_value + current_epf_value + current_ppf_value
inflation_adjusted_total_corpus = inflation_adjust(total_current_value, ms_inflation_rate, ms_years)
nps_contribution_percent = (current_nps_value / total_current_value) * 100 if total_current_value > 0 else 0
epf_contribution_percent = (current_epf_value / total_current_value) * 100 if total_current_value > 0 else 0
ppf_contribution_percent = (current_ppf_value / total_current_value) * 100 if total_current_value > 0 else 0

summary_data = {
    "Investment Component": ["NPS", "EPF", "PPF", "Total"],
    "Total Invested": [total_nps_invested, total_epf_invested, total_ppf_invested, total_invested],
    "Current Value (Future ₹)": [current_nps_value, current_epf_value, current_ppf_value, total_current_value],
    "Contribution to Total Corpus (%)": [nps_contribution_percent, epf_contribution_percent, ppf_contribution_percent, 100 if total_current_value > 0 else 0],
    "Inflation Adjusted Value (Today's ₹)": [nps_result["Total Corpus"][2], epf_inflation_adjusted_value, ppf_inflation_adjusted_value, inflation_adjusted_total_corpus]
}
summary_df = pd.DataFrame(summary_data)
currency_cols = ["Total Invested", "Current Value (Future ₹)", "Inflation Adjusted Value (Today's ₹)"]
for col in currency_cols:
    summary_df[col] = summary_df[col].apply(lambda x: f"₹{x:,.2f} ({toWords(x)})")
summary_df["Contribution to Total Corpus (%)"] = summary_df["Contribution to Total Corpus (%)"].apply(lambda x: f"{x:.2f}%")
st.dataframe(summary_df, use_container_width=True)

# ---------- AI Recommendations ----------
st.header("🧠 AI Recommendations")
st.warning("**Note:** These suggestions are for educational purposes only.\n\nAlways consult a qualified financial advisor before making investment decisions.")

if not st.session_state.get("rec_button_clicked", False) and "recommendations" not in st.session_state:
    if st.button("Get AI Recommendations", key="get_recs_btn"):
        st.session_state["rec_button_clicked"] = True
        st.rerun()

if st.session_state.get("rec_button_clicked", False) and "recommendations" not in st.session_state:
    with st.spinner("Fetching recommendations..."):
        try:
            backend_url = "https://corpus-compass.vercel.app/recommend"
            prompt = f"""
            Analyze the following retirement and investment calculations and provide actionable recommendations to help the user maximize their corpus and achieve their financial goals. Focus on the following:

            Master Inputs:
            - Years to Retirement: {ms_years}
            - Annual Increase in Investment: {ms_annual_increase:.2f}%
            - Inflation Rate: {ms_inflation_rate:.2f}%

            NPS Results:
            - Total Corpus: ₹{nps_result['Total Corpus'][0]:,.2f} ({nps_result['Total Corpus'][1]})
            - Inflation Adjusted Total Corpus: ₹{nps_result['Total Corpus'][2]:,.2f} ({nps_result['Total Corpus'][3]})
            - Monthly Pension: ₹{nps_result['Monthly Pension'][0]:,.2f} ({nps_result['Monthly Pension'][1]})
            - Inflation Adjusted Monthly Pension: ₹{nps_result['Monthly Pension'][2]:,.2f} ({nps_result['Monthly Pension'][3]})
            - Target Monthly Pension (Today's ₹): ₹{nps_result['Pension Gap Analysis']["Target Monthly Pension (Today's ₹)"] if 'Pension Gap Analysis' in nps_result else 'N/A':,.2f}
            - Pension Shortfall (Future ₹): ₹{nps_result['Pension Gap Analysis']['Pension Shortfall (Future ₹)'] if 'Pension Gap Analysis' in nps_result else 'N/A':,.2f}
            - Extra Monthly Investment Needed: ₹{nps_result['Pension Gap Analysis']['Extra Monthly Investment Needed'] if 'Pension Gap Analysis' in nps_result else 'N/A':,.2f} ({nps_result['Pension Gap Analysis']['Words']['Extra Monthly Investment Needed'] if 'Pension Gap Analysis' in nps_result else 'N/A'})

            EPF Results:
            - Future Value of Corpus: ₹{epf_future_value:,.2f} ({toWords(epf_future_value)})
            - Inflation Adjusted Value: ₹{epf_inflation_adjusted_value:,.2f} ({toWords(epf_inflation_adjusted_value)})

            PPF Results:
            - Future Value of Corpus: ₹{ppf_future_value:,.2f} ({toWords(ppf_future_value)})
            - Inflation Adjusted Value: ₹{ppf_inflation_adjusted_value:,.2f} ({toWords(ppf_inflation_adjusted_value)})

            Investment Summary:
            - Total Invested: ₹{total_invested:,.2f} ({toWords(total_invested)})
            - Total Current Value: ₹{total_current_value:,.2f} ({toWords(total_current_value)})
            - Inflation Adjusted Total Corpus: ₹{inflation_adjusted_total_corpus:,.2f} ({toWords(inflation_adjusted_total_corpus)})
            - NPS Contribution to Total Corpus: {nps_contribution_percent:.2f}%
            - EPF Contribution to Total Corpus: {epf_contribution_percent:.2f}%
            - PPF Contribution to Total Corpus: {ppf_contribution_percent:.2f}%

            Instructions:
            - You must provide the output as a bulleted list only, max 6 bullet points.
            - Do not write any introductory sentences, analysis, headers, or concluding remarks.
            - The response must begin directly with the first bullet point (*).
            - Provide recommendations in a clear, bullet-point format only.
            - Ensure each recommendation is concise and directly actionable.
            - Base the recommendations on the provided data.
            - If there is a Pension Shortfall:
                - Suggest specific actions to increase investments to cover the Extra Monthly Investment Needed.
                - Recommend adjustments to the investment allocation to potentially increase returns.
                - Advise on the feasibility of the Annual Increase in Investment.
            - If the Target Monthly Pension is met or exceeded:
                - Suggest strategies for further wealth maximization and diversification beyond the current instruments.
                - Propose considering an earlier retirement timeline.
                - Recommend ways to optimize the existing investment portfolio for higher returns or lower risk.
            - Provide general long-term financial planning tips relevant to the user's situation.
            """
            response = requests.post(backend_url, json={"prompt": prompt}, timeout=60)
            if response.status_code == 200:
                st.session_state["recommendations"] = response.json()["recommendations"]
            else:
                st.error(f"Failed: {response.json().get('error', response.text)}")
        except Exception as e:
            st.error(f"Error contacting backend: {e}")

if "recommendations" in st.session_state:
    st.markdown(st.session_state["recommendations"])
