import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

from main import (
    BuyAssumptions,
    RentAssumptions,
    GlobalAssumptions,
    Simulator,
    Mortgage,
)

st.set_page_config(page_title="Rent vs Buy Dashboard", layout="wide")
st.title("🏠 Rent vs Buy — Interactive Dashboard")

intro = st.markdown(
    """
    **What this is:** an interactive calculator to compare renting vs buying the same home.

    **How it works:** we simulate monthly cashflows for both paths (mortgage payments, VvE, maintenance, taxes; versus rent and investing the saved upfront cash), then compute **NPV** and **IRR**.

    **NPV (Net Present Value):** all future cashflows are discounted back to today using your **discount rate**, so euros at different times are comparable. Higher NPV = financially better.

    Use the sidebar to change assumptions; the summary and plots update live.
    """
)

# Placeholder for a one-line summary that we fill after running the sim
summary_placeholder = st.empty()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    st.subheader("Global")
    horizon_years = st.slider("Horizon (years)", 1, 30, 7, help="How long you expect to keep the house / stay in the rental before selling or moving.")
    discount_rate = st.number_input("Discount rate (annual)", value=0.03, min_value=0.0, max_value=0.25, step=0.005, format="%.3f", help="Your opportunity cost of capital (required return). Used to discount future cashflows into today's euros. Typical 2–5%.")

    st.subheader("Buy")
    price = st.number_input("House price", value=500_000, min_value=50_000, step=10_000, help="Purchase price of the property.")
    down_payment_pct = st.slider("Down payment (%)", 0, 60, 20, help="Percent of the price you pay upfront; the rest is financed.") / 100.0
    mortgage_rate = st.number_input("Mortgage rate (annual)", value=0.0361, min_value=0.0, max_value=1.0, step=0.001, format="%.4f", help="Nominal annual interest rate of the mortgage (e.g., 0.0361 = 3.61%).")
    term_years = st.slider("Mortgage term (years)", 5, 40, 30, help="Amortization length. Payment is computed for this full term.")
    vve_monthly = st.number_input("VvE (HOA) per month", value=200, min_value=0, step=25, help="Monthly owners' association fee (VvE/HOA). Often includes building insurance.")
    maint_pct = st.number_input("Maintenance (%% of value / year)", value=0.005, min_value=0.0, max_value=0.05, step=0.001, format="%.3f", help="Annual upkeep as a percent of current property value. 0.5% is common for flats in good condition.")
    property_tax_annual = st.number_input("Property tax (annual)", value=600, min_value=0, step=50, help="Municipal tax (e.g., OZB) per year. Set to 0 if you don't want to include it.")

    transfer_tax_pct = st.number_input("Transfer tax (%)", value=0.02, min_value=0.0, max_value=0.10, step=0.005, format="%.3f", help="One‑off tax on purchase (e.g., NL ~2% for owner‑occupiers; check your eligibility).")
    buyer_closing_costs_pct = st.number_input("Buyer closing costs (%)", value=0.02, min_value=0.0, max_value=0.08, step=0.005, format="%.3f", help="Notary, registration, valuation, bank fees, etc., as a % of price.")
    seller_costs_pct = st.number_input("Seller costs (%)", value=0.015, min_value=0.0, max_value=0.05, step=0.005, format="%.3f", help="Costs when selling (agent fees, notary), as a % of final sale price.")

    appreciation = st.number_input("Annual appreciation", value=0.02, min_value=-0.1, max_value=0.20, step=0.005, format="%.3f", help="Expected average annual change in the home's market value.")

    st.subheader("Rent")
    monthly_rent = st.number_input("Monthly rent (start)", value=2000, min_value=0, step=50, help="Your initial monthly rent at month 1.")
    rent_growth = st.number_input("Annual rent increase", value=0.05, min_value=0.0, max_value=0.20, step=0.005, format="%.3f", help="Expected average annual rent growth.")
    invest_return = st.number_input("Annual investment return (opportunity)", value=0.03, min_value=0.0, max_value=0.25, step=0.005, format="%.3f", help="Return you expect on money not spent buying (e.g., index fund). Used in the RENT path.")

# Assemble assumption objects
buy = BuyAssumptions(
    price=price,
    down_payment_pct=down_payment_pct,
    mortgage_rate=mortgage_rate,
    mortgage_term_years=term_years,
    vve_monthly=vve_monthly,
    maintenance_pct_per_year=maint_pct,
    property_tax_annual=property_tax_annual,
    transfer_tax_pct=transfer_tax_pct,
    buyer_closing_costs_pct=buyer_closing_costs_pct,
    seller_costs_pct=seller_costs_pct,
    annual_appreciation=appreciation,
)

rent = RentAssumptions(
    monthly_rent=monthly_rent,
    annual_rent_increase=rent_growth,
    renter_insurance_annual=0.0,
    annual_investment_return=invest_return,
)

global_ = GlobalAssumptions(
    horizon_years=horizon_years,
    discount_rate_annual=discount_rate,
)

sim = Simulator(buy, rent, global_)

# Run once for summary
buy_res = sim.simulate_buy()
rent_res = sim.simulate_rent()

# Helper formatters
fmt_money = lambda x: f"€{x:,.0f}".replace(",", "_").replace("_", ",")
fmt_pct = lambda x: f"{x*100:.2f}%"

# -----------------------------
# QUICK INPUTS SUMMARY
# -----------------------------
# Derived numbers
_dp_amt = price * down_payment_pct
_mortgage_principal = buy.mortgage_principal()
_m = Mortgage(_mortgage_principal, mortgage_rate, term_years)
_pmt = _m.payment()
_upfront = buy.upfront_cash_needed()

st.subheader("Your inputs (quick glance)")
colL, colR = st.columns(2)
with colL:
    st.markdown(
        f"""
        **Horizon:** {horizon_years} years  
        **Discount rate:** {fmt_pct(discount_rate)}  
        **Home price:** {fmt_money(price)}  
        **Down payment:** {fmt_money(_dp_amt)} ({fmt_pct(down_payment_pct)})  
        **Mortgage:** {fmt_pct(mortgage_rate)} × {term_years}y → **{fmt_money(_pmt)}/mo**  
        **Upfront (down + taxes/fees):** {fmt_money(_upfront)}
        """
    )
with colR:
    st.markdown(
        f"""
        **Rent (start):** {fmt_money(monthly_rent)}/mo  
        **Rent increase:** {fmt_pct(rent_growth)}  
        **Investment return:** {fmt_pct(invest_return)}  
        **VvE (HOA):** {fmt_money(vve_monthly)}/mo  
        **Maintenance:** {fmt_pct(maint_pct)}/yr of value  
        **Property tax:** {fmt_money(property_tax_annual)}/yr  
        **Transfer tax:** {fmt_pct(transfer_tax_pct)}  
        **Buyer closing:** {fmt_pct(buyer_closing_costs_pct)}  
        **Seller costs:** {fmt_pct(seller_costs_pct)}  
        **Appreciation:** {fmt_pct(appreciation)}
        """
    )

# =============================
# TOP SUMMARY CARDS
# =============================
# Compute delta early and show the one-line summary BEFORE the metrics
delta = buy_res.npv - rent_res.npv
verdict = "BUY" if delta > 0 else ("RENT" if delta < 0 else "TIE")
summary_placeholder.info(
    f"**Summary:** At your inputs, NPV(Buy) = {fmt_money(buy_res.npv)}, "
    f"NPV(Rent) = {fmt_money(rent_res.npv)}.\n\n"
    f"It is more convenient to **{verdict}** by {fmt_money(abs(delta))} (NPV)."
)

# -----------------------------
# CASH OUT (undiscounted) METRICS
# -----------------------------
cash_buy = buy_res.total_cash_out            # already net of sale proceeds
cash_rent = rent_res.total_rent_paid         # sum of rent over horizon (investment handled separately)
cash_delta = cash_buy - cash_rent

c1, c2, c3 = st.columns(3)
c1.metric("Total cash out (Buy)", fmt_money(-cash_buy))
c2.metric("Total cash out (Rent)", fmt_money(-cash_rent))
c3.metric("Cash out Δ (Buy − Rent)", fmt_money(-(cash_delta)))

st.caption("Total cash out is *undiscounted* net spending (negative = outflow). Buy is net of sale proceeds; Rent is rent paid. NPV differs because it discounts timing and includes all inflows/outflows in today-euros.")

col1, col2, col3 = st.columns(3)
col1.metric("NPV (Buy)", fmt_money(buy_res.npv))
col2.metric("NPV (Rent)", fmt_money(rent_res.npv))
col3.metric("NPV Δ (Buy − Rent)", fmt_money(delta), delta=None)

st.divider()

# =============================
# TEXT SUMMARY
# =============================
with st.expander("Detailed summary (same as CLI report)", expanded=False):
    st.code(sim.render_report())

st.divider()

# =============================
# PLOTS
# =============================

# 1) Cumulative cashflows over time (monthly)
st.subheader("Cumulative cashflows over time (monthly)")
months = global_.horizon_years * 12

buy_cum = np.cumsum(buy_res.cashflows)
rent_cum = np.cumsum(rent_res.cashflows)

# X spans from 0 (upfront) to M months (inclusive), so length = months+1
x_buy = np.arange(len(buy_cum))
x_rent = np.arange(len(rent_cum))

fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(x_buy, buy_cum, label="Buy — cumulative cashflow", linewidth=2)
ax1.plot(x_rent, rent_cum, label="Rent — cumulative cashflow", linewidth=2)
ax1.axhline(0, linewidth=1, color="black")
ax1.set_xlabel("Month (0 = upfront)")
ax1.set_ylabel("Cumulative cashflow (€)")
ax1.legend()
st.pyplot(fig1, use_container_width=False)
st.caption("Cumulative cashflows show the total outlay over time. Higher line = more money spent. Positive slope = cash outflows; jumps upward at the end = sale proceeds or investment payout.")

# 2) NPV vs horizon sweep
st.subheader("NPV vs Horizon sweep")
st.caption("Compares the net present value of buying vs renting at different horizons. If the Buy line is above the Rent line, buying is financially better at that horizon.")
max_h = st.slider("Max horizon for sweep (years)", 3, 40, max(10, horizon_years))
hs = list(range(1, max_h + 1))
buy_npvs, rent_npvs = [], []
for h in hs:
    s2 = Simulator(buy, rent, GlobalAssumptions(horizon_years=h, discount_rate_annual=discount_rate))
    br = s2.simulate_buy()
    rr = s2.simulate_rent()
    buy_npvs.append(br.npv)
    rent_npvs.append(rr.npv)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(hs, buy_npvs, label="Buy NPV", linewidth=2)
ax2.plot(hs, rent_npvs, label="Rent NPV", linewidth=2)
ax2.set_xlabel("Horizon (years)")
ax2.set_ylabel("NPV (€)")
ax2.legend()
st.pyplot(fig2, use_container_width=False)

# 3) Parameter sweep: Down payment vs horizon (NPV delta)
st.subheader("Heatmap: Down payment vs Horizon (NPV Buy − Rent)")
min_dp, max_dp = 0.0, 0.6
steps_dp = st.slider("Down payment grid steps", 3, 13, 7)
steps_h  = st.slider("Horizon grid steps", 3, 13, 7)

# Build grid
DP = np.linspace(min_dp, max_dp, steps_dp)
H  = np.linspace(1, max_h, steps_h, dtype=int)
Z  = np.zeros((len(H), len(DP)))
for i, h in enumerate(H):
    for j, dp in enumerate(DP):
        b2 = BuyAssumptions(
            price=price,
            down_payment_pct=float(dp),
            mortgage_rate=mortgage_rate,
            mortgage_term_years=term_years,
            vve_monthly=vve_monthly,
            maintenance_pct_per_year=maint_pct,
            property_tax_annual=property_tax_annual,
            transfer_tax_pct=transfer_tax_pct,
            buyer_closing_costs_pct=buyer_closing_costs_pct,
            seller_costs_pct=seller_costs_pct,
            annual_appreciation=appreciation,
        )
        s3 = Simulator(b2, rent, GlobalAssumptions(horizon_years=int(h), discount_rate_annual=discount_rate))
        br = s3.simulate_buy()
        rr = s3.simulate_rent()
        Z[i, j] = br.npv - rr.npv

fig3, ax3 = plt.subplots(figsize=(6, 4))
# Symmetric color scale around 0 so colors map consistently (blue = negative, red = positive)
vmax = float(np.max(np.abs(Z))) if np.max(np.abs(Z)) > 0 else 1.0
c = ax3.imshow(
    Z,
    aspect='auto',
    origin='lower',
    extent=[DP[0]*100, DP[-1]*100, H[0], H[-1]],
    cmap="coolwarm",
    vmin=-vmax,
    vmax=vmax,
)
# Break-even contour where NPV Δ = 0
X = np.linspace(DP[0]*100, DP[-1]*100, len(DP))
Y = np.linspace(H[0], H[-1], len(H))
ax3.contour(X, Y, Z, levels=[0], colors='black', linewidths=1)
fig3.colorbar(c, ax=ax3, label="NPV Δ (Buy − Rent) €")
ax3.set_xlabel("Down payment (%)")
ax3.set_ylabel("Horizon (years)")
st.pyplot(fig3, use_container_width=False)
st.caption("Heatmap shows how the relative advantage (Buy − Rent) changes with different down payment percentages and horizons. **Red = Buy better**, **Blue = Rent better**. The black line marks the break‑even (NPV Δ = 0).")

st.caption("Tip: Use the sidebar to change appreciation, rent growth, fees, and rates; then scan the sweep plots to see where the verdict flips.")

# 4) Mortgage amortization breakdown (interest vs principal)
st.subheader("Mortgage payment split over time (interest vs principal)")

# Add a slider to select number of months to display (up to horizon)
max_months = global_.horizon_years * 12
months_slider = st.slider("Months to display", 12, max_months, max_months, step=12)

m = Mortgage(buy.mortgage_principal(), mortgage_rate, term_years)
pmt = m.payment()
r = mortgage_rate / 12.0
balance = buy.mortgage_principal()
ints = []
prins = []
for _ in range(months_slider):
    interest = balance * r
    principal = max(0.0, pmt - interest)
    ints.append(interest)
    prins.append(principal)
    balance = max(0.0, balance - principal)

x_idx = np.arange(1, months_slider + 1)
figA, axA = plt.subplots(figsize=(6, 4))
axA.stackplot(x_idx, ints, prins, labels=["Interest", "Principal"])
axA.set_xlabel("Month")
axA.set_ylabel("Monthly payment split (€)")
axA.legend(
    # Put legend outside the plot area
    loc="upper left",
    bbox_to_anchor=(.5, 1.15),
    ncol=2, 
)
st.pyplot(figA, use_container_width=False)
st.caption("Shows how each monthly payment is divided. Early payments are interest-heavy; over time, the principal share grows.")


st.markdown("<hr />", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size: 0.9em;'>Built by Carlo — self-hosted on Raspberry Pi 🥔🚀</div>", unsafe_allow_html=True)
