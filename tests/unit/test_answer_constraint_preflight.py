from x_creative.answer.constraint_preflight import preflight_user_constraints
from x_creative.core.types import ProblemFrame


def test_preflight_preserves_numbered_user_constraints_from_raw_question() -> None:
    problem = ProblemFrame(
        description=(
            "寻找一个适用于中国大陆 A 股市场的板块轮动策略架构，要求：\n"
            "1. 能够适应不同市场 regime 的切换，\n"
            "2. 可用数据只有来自于 tushare 的个股日线、ETF日线、财报、券商研报、融资融券、可转债，以及来自于淘宝的个股逐笔成交数据，\n"
            "3. 满足涨停无法买入跌停无法卖出、T+1交割的交易规则。"
        ),
        constraints=[
            "Data sources are restricted to Tushare (daily/financial/margin/reports) and Taobao (tick-level transaction data)."
        ],
    )

    updated = preflight_user_constraints(problem)

    assert any("可转债" in text for text in updated.constraints)
    assert any("可转债" in spec.text for spec in updated.structured_constraints)
