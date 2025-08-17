import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai

# ---------- Finance math functions (Copied from the notebook) ----------
def _trim_zeros(s: str) -> str:
    """Trim trailing zeros and decimal point."""
    s = s.rstrip('0').rstrip('.')
    return s if s else "0"

def toWords(num: float, decimals: int = 2) -> str:
    """
    Compact Indian numbering:
    - >= 1 Cr.  (1e7)
    - >= 1 Lakh (1e5)
    - >= 1 k    (1e3)
    Rounds to `decimals` and trims trailing zeros.
    Handles negatives.
    """
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
        # For < 1k, show rounded integer
        text = str(int(round(n)))
        return f"-{text}" if neg and text != "0" else text

    text = _trim_zeros(f"{val:.{decimals}f}")
    return f"-{text} {unit}" if neg else f"{text} {unit}"

def inflation_adjust(value: float, inflation_rate: float, years: int) -> float:
    """Convert a future-value amount into today's money."""
    return value / ((1 + inflation_rate) ** years)

def inflate_future(value: float, inflation_rate: float, years: int) -> float:
    """Inflate today's money into future value."""
    return value * ((1 + inflation_rate) ** years)

def fv_growing_annuity_monthly(monthly_pmt: float, g: float, r: float, n_years: int) -> float:
    """
    Future value of a growing annuity with monthly payments growing at g annually
    and compounding at r annually.
    """
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

# ---------- Calculator functions (Copied from the notebook) ----------
def nps_calculator(monthly_investment: float, annual_increase: float, years: int, expected_return: float) -> float:
    """Wrapper: FV of growing annuity with monthly payments."""
    return fv_growing_annuity_monthly(monthly_investment, annual_increase, expected_return, years)

def nps_withdrawal_pension(
    monthly_investment: float,
    annual_increase: float,
    years: int,
    expected_return: float,
    withdrawal_percent: float,
    annuity_return: float,
    life_expectancy: int = 20 * 12,   # months
    inflation_rate: float = 0.06,
    target_pension_today: float | None = None,
    words_decimals: int = 2
):
    """
    Computes:
      - Total corpus at retirement
      - Lumpsum and annuity corpus (based on withdrawal_percent)
      - Monthly pension from annuity corpus
      - Inflation-adjusted (today's money) versions
      - Pension gap vs. a target stated in today's rupees, along with extra investment needed
    """
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
    """
    Calculates the future value of EPF investments with a yearly increase in investment.
    """
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
    """
    Calculates the future value of PPF investments.
    """
    max_annual_investment = 150000
    annual_investment = min(annual_investment, max_annual_investment)

    future_value = 0
    for year in range(years):
        fv_this_year = annual_investment * ((1 + annual_interest_rate) ** (years - (year + 1)))
        future_value += fv_this_year

    return future_value

# ---------- Streamlit App ----------
st.title("Comprehensive Retirement and Investment Calculators")

st.markdown("""
Welcome to the Retirement and Investment Calculators! This tool helps you estimate the potential future value of your investments in NPS, EPF, and PPF based on your inputs, and understand the impact of inflation.

**How to use:**
1. Adjust the input parameters in the sidebar on the left.
2. The results for each calculator and a summary of your total investments will update automatically below.
3. Review the "Calculation Results" and "Investment Summary" sections to see your projected corpus and pension.
""")

st.header("Table of Contents")
st.markdown("""
- [Master Inputs](#master-inputs)
- [NPS Inputs](#nps-inputs)
- [EPF Inputs](#epf-inputs)
- [PPF Inputs](#ppf-inputs)
- [Calculation Results](#calculation-results)
  - [NPS Calculation](#nps-calculation)
  - [EPF Calculation](#epf-calculation)
  - [PPF Calculation](#ppf-calculation)
- [Investment Summary](#investment-summary)
- [AI Recommendations](#ai-recommendations)
""")

st.header("Master Inputs")
st.write("These inputs apply to all calculations and represent your overall retirement planning horizon and economic assumptions.")
ms_years = st.number_input("Years to Retirement", min_value=1, value=37, help="Enter the number of years until you plan to retire.")
ms_annual_increase = st.number_input("Annual Increase in Investment (%)", min_value=0.0, value=10.0, help="Enter the expected annual percentage increase in your investment amounts.") / 100.0
ms_inflation_rate = st.number_input("Inflation Rate (%)", min_value=0.0, value=7.0, help="Enter the expected average annual inflation rate. This is used to calculate future values in today's money.") / 100.0

# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.header("NPS Inputs")
    st.write("Specify the details for your NPS investments.")
    ms_nps_annual_investment = st.number_input("NPS Annual Investment (₹)", min_value=0, value=96589, help="Enter your annual investment amount in NPS.")
    ms_nps_expected_return = st.number_input("NPS Expected Return (%)", min_value=0.0, value=12.0, help="Enter the expected annual return percentage on your NPS investments during the accumulation phase.") / 100.0
    ms_nps_lumpsum_withdrawal_percent = st.number_input("NPS Lumpsum Withdrawal (%)", min_value=0.0, max_value=100.0, value=0.0, help="Enter the percentage of your NPS corpus you plan to withdraw as a lumpsum at retirement.") / 100.0
    ms_nps_annuity_return = st.number_input("NPS Annuity Return (%)", min_value=0.0, value=7.0, help="Enter the expected annual return percentage from the annuity purchased with the remaining NPS corpus.") / 100.0
    ms_nps_life_expectancy = st.number_input("NPS Life Expectancy (Years for Pension)", min_value=1, value=20, help="Enter the number of years you expect to receive pension payments from the annuity.")
    ms_targeted_pension_today = st.number_input("Target Monthly Pension (Today's ₹)", min_value=0, value=100000, help="Enter your desired monthly pension amount in today's rupees.")

with col2:
    st.header("EPF Inputs")
    st.write("Specify the details for your EPF contributions.")
    ms_epf_monthly_investment = st.number_input("EPF Monthly Investment (₹)", min_value=0, value=15000, help="Enter your monthly contribution to EPF.")
    ms_epf_annual_interest_rate = st.number_input("EPF Annual Interest Rate (%)", min_value=0.0, value=8.25, help="Enter the current annual interest rate for EPF.") / 100.0

    st.header("PPF Inputs")
    st.write("Specify the details for your PPF investments.")
    ms_ppf_annual_investment = st.number_input("PPF Annual Investment (₹)", min_value=0, value=150000, help="Enter your annual investment amount in PPF. Note: PPF has a maximum annual investment limit.")
    ms_ppf_annual_interest_rate = st.number_input("PPF Annual Interest Rate (%)", min_value=0.0, value=7.1, help="Enter the current annual interest rate for PPF.") / 100.0


# Perform Calculations
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


# Display Results
st.header("Calculation Results")

st.subheader("NPS Calculation")
st.write("Below are the projected values for your NPS investment at retirement:")
nps_summary_data = {
    "Metric": [k for k in nps_result if k != "Pension Gap Analysis"],
    "Future Value (₹)": [v[0] for k,v in nps_result.items() if k != "Pension Gap Analysis"],
    "Future Value": [v[1] for k,v in nps_result.items() if k != "Pension Gap Analysis"],
    "Inflation Adjusted Value (Today's ₹)": [v[2] for k,v in nps_result.items() if k != "Pension Gap Analysis"],
    "Inflation Adjusted Value": [v[3] for k,v in nps_result.items() if k != "Pension Gap Analysis"]
}
nps_summary_df = pd.DataFrame(nps_summary_data)
st.dataframe(nps_summary_df)

if "Pension Gap Analysis" in nps_result:
    st.subheader("NPS Pension Gap Analysis")
    st.write("This section compares your projected monthly pension from NPS with your target pension in today's rupees and estimates any shortfall and the extra investment needed.")
    gap_data = nps_result["Pension Gap Analysis"]["Words"]
    gap_summary_data = {
        "Metric": [k for k in gap_data],
        "Value": [v for k,v in gap_data.items()]
    }
    gap_summary_df = pd.DataFrame(gap_summary_data)
    st.dataframe(gap_summary_df)


st.subheader("EPF Calculation")
st.write("Below is the projected future value of your EPF corpus at retirement:")
epf_summary_data = {
    "Metric": ["Future Value of EPF Corpus", "Inflation Adjusted EPF Value"],
    "Value (₹)": [epf_future_value, epf_inflation_adjusted_value],
    "Value": [toWords(epf_future_value), toWords(epf_inflation_adjusted_value)]
}
epf_summary_df = pd.DataFrame(epf_summary_data)
st.dataframe(epf_summary_df)


st.subheader("PPF Calculation")
st.write("Below is the projected future value of your PPF corpus at retirement:")
ppf_summary_data = {
    "Metric": ["Future Value of PPF", "Inflation Adjusted PPF Value"],
    "Value (₹)": [ppf_future_value, ppf_inflation_adjusted_value],
    "Value": [toWords(ppf_future_value), toWords(ppf_inflation_adjusted_value)]
}
ppf_summary_df = pd.DataFrame(ppf_summary_data)
st.dataframe(ppf_summary_df)


st.header("Investment Summary")
st.write("This table provides a consolidated summary of your investments across NPS, EPF, and PPF, showing the total invested amount, projected future value, contribution percentage of each component to the total corpus, and the inflation-adjusted value in today's rupees.")


# Calculate total invested amount (with annual increase for NPS and EPF)
total_nps_invested = sum((ms_nps_annual_investment / 12) * ((1 + ms_annual_increase) ** year) * 12 for year in range(ms_years))
total_epf_invested = sum(ms_epf_monthly_investment * ((1 + ms_annual_increase) ** year) * 12 for year in range(ms_years))
total_ppf_invested = min(ms_ppf_annual_investment, 150000) * ms_years # PPF is capped and no annual increase in limit


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

st.dataframe(summary_df)

# ---------- AI Recommendations ----------
st.header("AI Recommendations")
st.write("Based on your inputs and calculated results, here are some financial recommendations. Please remember that these recommendations are for educational purposes only and should not be considered financial advice. Consult with a qualified financial advisor for personalized guidance.")

# Configure the generative AI model
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash-lite') # Using a suitable model for text generation
except Exception as e:
    st.error(f"Failed to configure the AI model. Please check your GOOGLE_API_KEY in Colab secrets. Error: {e}")
    model = None # Set model to None if configuration fails


if model:
    # Prepare the prompt for the AI model
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

    try:
        with st.spinner("Generating recommendations..."):
            response = model.generate_content(prompt)
            st.markdown(response.text)
    except Exception as e:
        st.error(f"An error occurred while generating recommendations: {e}")
else:
    st.warning("AI recommendations are not available because the AI model could not be configured.")