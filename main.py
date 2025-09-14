from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

############################################################
# Pretty-print / reporting utilities
############################################################

def _fmt_money(x: float, currency: str = "€") -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}{currency}{abs(x):,.0f}".replace(",", "_").replace("_", ",")


def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _fmt_rate(x: float) -> str:
    return _fmt_pct(x)


def _fmt_num(x: float) -> str:
    return f"{x:,.0f}".replace(",", "_").replace("_", ",")


def _hr(char: str = "─", n: int = 60) -> str:
    return char * n


def _twocol_table(rows):
    # rows: List[Tuple[str, str, str]] -> label, buy, rent
    left_w = max(len(r[0]) for r in rows) if rows else 0
    mid_w = max(len(r[1]) for r in rows) if rows else 0
    out_lines = []
    header = f"{'Metric'.ljust(left_w)}  {'Buy'.rjust(mid_w)}    Rent"
    out_lines.append(header)
    out_lines.append(_hr())
    for label, a, b in rows:
        out_lines.append(f"{label.ljust(left_w)}  {a.rjust(mid_w)}    {b}")
    return "\n".join(out_lines)

############################################################
# Core financial utilities
############################################################

def annuity_payment(principal: float, annual_rate: float, years: int, payments_per_year: int = 12) -> float:
    """Monthly (or periodic) payment for an amortizing loan.
    principal: loan amount
    annual_rate: nominal annual interest rate (e.g. 0.04 for 4%)
    years: term in years
    payments_per_year: usually 12
    """
    r = annual_rate / payments_per_year
    n = years * payments_per_year
    if annual_rate == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def present_value(cashflows: List[float], rate: float, payments_per_year: int = 12) -> float:
    """Net present value of periodic cashflows occurring at the end of each period.
    rate is nominal annual rate; internally converted to per-period.
    """
    r = rate / payments_per_year
    return sum(cf / ((1 + r) ** t) for t, cf in enumerate(cashflows, start=1))


def internal_rate_of_return(cashflows: List[float], payments_per_year: int = 12, guess: float = 0.05) -> Optional[float]:
    """Simple IRR via bisection on nominal annual rate. Returns None if sign does not change.
    cashflows: list of periodic cashflows (end of period). Typically starts negative.
    """
    # Ensure there is a sign change
    if all(cf >= 0 for cf in cashflows) or all(cf <= 0 for cf in cashflows):
        return None

    # Bisection bounds for annual rate
    low, high = -0.99, 1.50  # -99% to +150% nominal per year

    def npv_at(rate: float) -> float:
        return present_value(cashflows, rate=rate, payments_per_year=payments_per_year)

    npv_low = npv_at(low)
    npv_high = npv_at(high)
    if npv_low * npv_high > 0:
        return None

    for _ in range(100):
        mid = (low + high) / 2
        npv_mid = npv_at(mid)
        if abs(npv_mid) < 1e-8:
            return mid
        if npv_low * npv_mid < 0:
            high, npv_high = mid, npv_mid
        else:
            low, npv_low = mid, npv_mid
    return (low + high) / 2


############################################################
# Mortgage
############################################################

@dataclass
class Mortgage:
    principal: float
    annual_rate: float
    term_years: int
    payments_per_year: int = 12

    def payment(self) -> float:
        return annuity_payment(self.principal, self.annual_rate, self.term_years, self.payments_per_year)

    def balance_after(self, months: int) -> float:
        """Remaining balance after `months` payments."""
        pmt = self.payment()
        r = self.annual_rate / self.payments_per_year
        n = self.term_years * self.payments_per_year
        m = months
        if self.annual_rate == 0:
            paid_principal = pmt * m
            return max(0.0, self.principal - paid_principal)
        return self.principal * ((1 + r) ** n - (1 + r) ** m) / ((1 + r) ** n - 1)

    def interest_principal_split(self, months: int) -> Tuple[float, float]:
        """Total interest and principal paid over the first `months` payments."""
        pmt = self.payment()
        bal = self.principal
        r = self.annual_rate / self.payments_per_year
        total_interest = 0.0
        total_principal = 0.0
        for _ in range(months):
            interest = bal * r
            principal = pmt - interest
            total_interest += interest
            total_principal += principal
            bal -= principal
            if bal <= 1e-8:
                break
        return total_interest, total_principal


############################################################
# Assumptions (inputs)
############################################################

@dataclass
class BuyAssumptions:
    price: float
    down_payment_pct: float  # e.g. 0.20 for 20%
    mortgage_rate: float     # nominal annual
    mortgage_term_years: int

    # Recurring costs
    vve_monthly: float = 0.0                 # owners' association (VvE) fees
    maintenance_pct_per_year: float = 0.005  # heuristic: 0.5% of property value per year (was 1%)
    property_tax_annual: float = 0.0         # OZB or equivalent

    # One-off transaction costs (purchase)
    transfer_tax_pct: float = 0.0            # e.g. NL: 2% for owner-occupiers (check rules)
    buyer_closing_costs_pct: float = 0.02    # notary, valuation, registry, bank fee etc.

    # Selling costs
    seller_costs_pct: float = 0.02           # agent fees, notary, etc.

    # Price dynamics
    annual_appreciation: float = 0.02        # expected growth
    selling_price_override: Optional[float] = None  # if set, overrides appreciation model

    def upfront_cash_needed(self) -> float:
        dp = self.price * self.down_payment_pct
        closing = self.price * (self.transfer_tax_pct + self.buyer_closing_costs_pct)
        return dp + closing

    def mortgage_principal(self) -> float:
        return self.price * (1 - self.down_payment_pct)


@dataclass
class RentAssumptions:
    monthly_rent: float
    annual_rent_increase: float = 0.02
    renter_insurance_annual: float = 0.0

    # Opportunity cost: invest what would have been the buyer's upfront costs
    annual_investment_return: float = 0.03


@dataclass
class GlobalAssumptions:
    horizon_years: int
    inflation_annual: float = 0.0  # if >0, results can be shown in real terms
    discount_rate_annual: float = 0.03  # used for NPV


############################################################
# Results dataclasses
############################################################

@dataclass
class BuyResult:
    total_cash_out: float
    net_equity_at_sale: float
    sale_proceeds_after_costs: float
    outstanding_mortgage_at_sale: float
    cashflows: List[float]  # negative for outflows, positive inflows (monthly)
    npv: float
    irr: Optional[float]


@dataclass
class RentResult:
    total_rent_paid: float
    investment_value_end: float
    cashflows: List[float]
    npv: float
    irr: Optional[float]


############################################################
# Simulator
############################################################

class Simulator:
    def __init__(self, buy: BuyAssumptions, rent: RentAssumptions, global_: GlobalAssumptions):
        self.buy = buy
        self.rent = rent
        self.global_ = global_

    # ---------------------- BUY PATH ---------------------- #
    def simulate_buy(self) -> BuyResult:
        months = self.global_.horizon_years * 12
        m_principal = self.buy.mortgage_principal()
        mortgage = Mortgage(m_principal, self.buy.mortgage_rate, self.buy.mortgage_term_years)
        pmt = mortgage.payment()

        # Upfront negative cashflow (down payment + closing)
        cashflows: List[float] = []
        upfront = self.buy.upfront_cash_needed()
        cashflows.append(-upfront)  # treat as occurring at end of month 1; conservative

        # Monthly recurring costs
        vve = self.buy.vve_monthly
        maintenance_monthly_base = (self.buy.price * self.buy.maintenance_pct_per_year) / 12
        property_tax_monthly = self.buy.property_tax_annual / 12

        # Simulate each month
        total_out = upfront
        for m in range(1, months + 1):
            # assume maintenance grows with price appreciation
            current_value = self._price_after_months(m)
            maintenance_monthly = (current_value * self.buy.maintenance_pct_per_year) / 12

            monthly_out = pmt + vve + property_tax_monthly + maintenance_monthly
            total_out += monthly_out
            cashflows.append(-monthly_out)

        # Sale at horizon
        sale_price = self._sale_price()
        outstanding = mortgage.balance_after(months)
        seller_costs = sale_price * self.buy.seller_costs_pct
        net_sale_proceeds = max(0.0, sale_price - seller_costs - outstanding)
        cashflows[-1] += net_sale_proceeds  # add to last period as positive inflow

        # Equity = sale price - selling costs - outstanding
        equity = max(0.0, sale_price - seller_costs - outstanding)

        # Financial metrics
        npv = present_value(cashflows, self.global_.discount_rate_annual)
        irr = internal_rate_of_return(cashflows)

        return BuyResult(
            total_cash_out=total_out - net_sale_proceeds,
            net_equity_at_sale=equity,
            sale_proceeds_after_costs=net_sale_proceeds,
            outstanding_mortgage_at_sale=outstanding,
            cashflows=cashflows,
            npv=npv,
            irr=irr,
        )

    def _price_after_months(self, months: int) -> float:
        years = months / 12
        return self.buy.price * ((1 + self.buy.annual_appreciation) ** years)

    def _sale_price(self) -> float:
        if self.buy.selling_price_override is not None:
            return self.buy.selling_price_override
        return self._price_after_months(self.global_.horizon_years * 12)

    # ---------------------- RENT PATH --------------------- #
    def simulate_rent(self) -> RentResult:
        months = self.global_.horizon_years * 12
        cashflows: List[float] = []

        # Upfront cash: in RENT, you allocate the same amount to an investment
        upfront_buy_cash = self.buy.upfront_cash_needed()
        cashflows.append(-upfront_buy_cash)  # treat as cash out allocated to investment

        # Track invested amount with monthly compounding at the assumed annual rate
        invest_balance = upfront_buy_cash
        invest_rate_monthly = self.rent.annual_investment_return / 12

        # Rent dynamics
        rent = self.rent.monthly_rent
        rent_increase_monthly = (1 + self.rent.annual_rent_increase) ** (1 / 12) - 1
        renter_insurance_monthly = self.rent.renter_insurance_annual / 12

        total_rent_paid = 0.0

        for _ in range(1, months + 1):
            # pay rent & insurance (outflows)
            out = rent + renter_insurance_monthly
            cashflows.append(-out)
            total_rent_paid += out

            # investment grows monthly
            invest_balance *= (1 + invest_rate_monthly)

            # rent grows monthly
            rent *= (1 + rent_increase_monthly)

        # At the end, capture the investment liquidation as a positive inflow
        cashflows[-1] += invest_balance

        npv = present_value(cashflows, self.global_.discount_rate_annual)
        irr = internal_rate_of_return(cashflows)

        return RentResult(
            total_rent_paid=total_rent_paid,
            investment_value_end=invest_balance,
            cashflows=cashflows,
            npv=npv,
            irr=irr,
        )

    # ---------------------- COMPARISON -------------------- #
    def compare(self) -> Dict[str, float]:
        buy_res = self.simulate_buy()
        rent_res = self.simulate_rent()

        return {
            "horizon_years": self.global_.horizon_years,
            "buy_total_net_outlay": buy_res.total_cash_out,  # out-of-pocket net of sale proceeds
            "buy_equity_at_sale": buy_res.net_equity_at_sale,
            "buy_npv": buy_res.npv,
            "buy_irr": buy_res.irr if buy_res.irr is not None else float("nan"),
            "rent_total_rent_paid": rent_res.total_rent_paid,
            "rent_investment_end": rent_res.investment_value_end,
            "rent_npv": rent_res.npv,
            "rent_irr": rent_res.irr if rent_res.irr is not None else float("nan"),
            "npv_delta_buy_minus_rent": buy_res.npv - rent_res.npv,
        }

    def render_report(self) -> str:
        months = self.global_.horizon_years * 12

        # Run sims once for NPVs/IRRs and cashflow-accurate sale/invest values
        buy_res = self.simulate_buy()
        rent_res = self.simulate_rent()

        # Rebuild mortgage and core params
        mortgage = Mortgage(self.buy.mortgage_principal(), self.buy.mortgage_rate, self.buy.mortgage_term_years)
        pmt = mortgage.payment()
        sale_price = self._sale_price()

        # ---------- BUY: compute monthly averages & cumulative totals (same logic as simulate_buy) ----------
        vve = self.buy.vve_monthly
        property_tax_monthly = self.buy.property_tax_annual / 12

        buy_total_monthly_out = 0.0
        buy_total_maint = 0.0
        for m in range(1, months + 1):
            current_value = self._price_after_months(m)
            maint_m = (current_value * self.buy.maintenance_pct_per_year) / 12
            buy_total_maint += maint_m
            buy_total_monthly_out += pmt + vve + property_tax_monthly + maint_m
        buy_avg_monthly_out = buy_total_monthly_out / months if months else 0.0

        # totals
        buy_total_mortgage_paid = pmt * months
        buy_int_paid, buy_prin_paid = mortgage.interest_principal_split(months)
        buy_total_vve = vve * months
        buy_total_tax = property_tax_monthly * months
        buy_upfront = self.buy.upfront_cash_needed()
        buy_selling_costs = sale_price * self.buy.seller_costs_pct

        # ---------- RENT: compute averages & totals ----------
        renter_ins_m = self.rent.renter_insurance_annual / 12
        # walk rent path deterministically (same as simulate_rent growth)
        rent_level = self.rent.monthly_rent
        rent_increase_m = (1 + self.rent.annual_rent_increase) ** (1 / 12) - 1
        rent_total_rent = 0.0
        for _ in range(1, months + 1):
            rent_total_rent += rent_level + renter_ins_m
            rent_level *= (1 + rent_increase_m)
        rent_avg_monthly_out = rent_total_rent / months if months else 0.0
        rent_total_ins = renter_ins_m * months
        rent_initial_cash_invested = self.buy.upfront_cash_needed()  # positive (retained & invested)

        title = "RENT vs BUY — Summary"
        lines = [
            title,
            _hr("="),
            f"Horizon: {self.global_.horizon_years} years | Discount rate: {_fmt_rate(self.global_.discount_rate_annual)}",
            f"House price: {_fmt_money(self.buy.price)} | Down payment: {_fmt_pct(self.buy.down_payment_pct)} ({_fmt_money(self.buy.price * self.buy.down_payment_pct)})",
            f"Mortgage: {_fmt_rate(self.buy.mortgage_rate)} for {self.buy.mortgage_term_years} years | Payment: {_fmt_money(pmt)}/mo",
            f"Assumed sale price at horizon: {_fmt_money(sale_price)}",
            _hr(),
        ]

        # High-level comparison table
        rows = [
            ("Upfront cash", _fmt_money(-buy_upfront), _fmt_money(-rent_initial_cash_invested) + " (invested)"),
            ("Avg monthly out", _fmt_money(buy_avg_monthly_out), _fmt_money(rent_avg_monthly_out)),
            ("Total cash out (net)", _fmt_money(buy_res.total_cash_out), _fmt_money(rent_res.total_rent_paid)),
            ("Outstanding mortgage @sale", _fmt_money(buy_res.outstanding_mortgage_at_sale), "—"),
            ("Sale proceeds (net of costs)", _fmt_money(buy_res.sale_proceeds_after_costs), "—"),
            ("Equity at sale", _fmt_money(buy_res.net_equity_at_sale), "—"),
            ("NPV (discounted)", _fmt_money(buy_res.npv), _fmt_money(rent_res.npv)),
            ("IRR (nominal, approx)", f"{buy_res.irr*100:.2f}%" if buy_res.irr is not None else "n/a", f"{rent_res.irr*100:.2f}%" if rent_res.irr is not None else "n/a"),
        ]
        lines.append(_twocol_table(rows))

        # Monthly breakdown block
        lines += [
            _hr(),
            "Monthly cash out — breakdown (averages over horizon):",
            f"  BUY   = mortgage {_fmt_money(pmt)} + VvE {_fmt_money(vve)} + maint ~{_fmt_money(buy_total_maint/months)} + tax {_fmt_money(property_tax_monthly)}",
            f"        ≈ {_fmt_money(buy_avg_monthly_out)} / month",
        ]
        rent_core_avg = rent_avg_monthly_out - renter_ins_m
        if renter_ins_m > 0:
            lines += [
                f"  RENT  = rent (avg over horizon) ~{_fmt_money(rent_core_avg)} + renter insurance {_fmt_money(renter_ins_m)}",
                f"        ≈ {_fmt_money(rent_avg_monthly_out)} / month",
            ]
        else:
            lines += [
                f"  RENT  = rent (avg over horizon) ~{_fmt_money(rent_core_avg)}",
                f"        ≈ {_fmt_money(rent_avg_monthly_out)} / month",
            ]

        # Cumulative cash summary block
        lines += [
            _hr(),
            "Cumulative cash summary over horizon:",
            "  BUY:",
            f"    Upfront (down + closing)                 {_fmt_money(buy_upfront)}",
            f"    Mortgage payments (total)                {_fmt_money(buy_total_mortgage_paid)}  [interest {_fmt_money(buy_int_paid)}, principal {_fmt_money(buy_prin_paid)}]",
            f"    VvE fees                                  {_fmt_money(buy_total_vve)}",
            f"    Maintenance                               {_fmt_money(buy_total_maint)}",
            f"    Property tax                               {_fmt_money(buy_total_tax)}",
            f"    Selling costs                              {_fmt_money(buy_selling_costs)}",
            f"    Sale proceeds (after costs)               {_fmt_money(buy_res.sale_proceeds_after_costs)}",
            f"    Net cash out (above inflows)              {_fmt_money(buy_res.total_cash_out)}",
            "  RENT:",
            f"    Initial cash invested (cash out)          {_fmt_money(rent_initial_cash_invested)}",
        ]
        if renter_ins_m > 0:
            lines += [f"    Rent + renter insurance (total)           {_fmt_money(rent_total_rent)}"]
        else:
            lines += [f"    Rent (total)                               {_fmt_money(rent_total_rent)}"]
        lines += [
            f"    Investment value at end                   {_fmt_money(rent_res.investment_value_end)}",
            f"    Net cash out (rent total only)            {_fmt_money(rent_res.total_rent_paid)}",
        ]

        # Verdict
        delta = buy_res.npv - rent_res.npv
        verdict = (
            f"BUY wins by {_fmt_money(abs(delta))} in NPV" if delta > 0 else
            f"RENT wins by {_fmt_money(abs(delta))} in NPV" if delta < 0 else
            "Tie on NPV"
        )
        lines += [
            _hr(),
            f"Verdict: {verdict}",
            _hr("="),
            "Notes:",
            "• \"Avg monthly out\" includes all recurring owner/renter costs, not just the mortgage.",
            "• Both paths include the same upfront cash outflow; in RENT it’s allocated to an investment that pays out at the end.",
            "• NPV discounts all monthly cashflows at your chosen rate so earlier/later euros are comparable.",
        ]
        return "\n".join(lines)

############################################################
# Example usage (you can delete this block later)
############################################################
if __name__ == "__main__":
    # --- Example assumptions (tweak to your case) ---

    buy = BuyAssumptions(
        price=500_000,                 # or set based on your down payment (see below)
        down_payment_pct=0.20,         # adjust to your plan
        mortgage_rate=0.0361,          # 3.61% effective
        mortgage_term_years=30,
        vve_monthly=200,
        maintenance_pct_per_year=0.001,
        property_tax_annual=600,
        transfer_tax_pct=0.02,
        buyer_closing_costs_pct=0.02,
        seller_costs_pct=0.015,
        annual_appreciation=0.02,
    )

    rent = RentAssumptions(
        monthly_rent=2_000,
        annual_rent_increase=0.045,
        renter_insurance_annual=0,
        annual_investment_return=0.03,
    )

    global_ = GlobalAssumptions(
        horizon_years=5,
        inflation_annual=0.0,
        discount_rate_annual=0.03,
    )

    sim = Simulator(buy, rent, global_)
    summary = sim.compare()

    print("\n" + sim.render_report())

    # You can also inspect detailed results:
    # buy_result = sim.simulate_buy()
    # rent_result = sim.simulate_rent()
