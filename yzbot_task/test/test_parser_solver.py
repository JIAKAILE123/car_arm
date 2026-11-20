from yzbot_task.command_parser import parse_command, format_command
from yzbot_task.local_solver import solve_xy_from_question, question_to_command
from yzbot_task.deepseek_client import _extract_xy_json
from yzbot_task.llm_solver import question_to_command_llm


def test_parse_bracket():
    tasks = parse_command("[red,4,A;blue,1,C]")
    assert len(tasks) == 2
    assert tasks[0].color == "red" and tasks[0].count == 4 and tasks[0].zone == "A"
    assert tasks[1].color == "blue" and tasks[1].count == 1 and tasks[1].zone == "C"


def test_parse_plain():
    tasks = parse_command("red,2,B")
    assert tasks[0].zone == "B"


def test_parse_cube_id():
    tasks = parse_command("[red#1,A;blue#2,B;red#3,C;blue#5,A;red#5,B]")
    assert len(tasks) == 5
    assert tasks[0].cube_id == 1 and tasks[0].color == "red" and tasks[0].zone == "A"
    assert tasks[1].cube_id == 2 and tasks[1].color == "blue"
    assert tasks[4].cube_id == 5 and tasks[4].color == "red" and tasks[4].zone == "B"


def test_solver_official_sample():
    q = (
        "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
        "小帅、小丽都需要7件T恤;小娜需要2件T恤,1条裤子;小杰需要1条裤子。"
        "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。"
    )
    x, y = solve_xy_from_question(q)
    assert (x, y) == (4, 1)
    cmd = question_to_command(q)
    assert cmd == "[red,4,A;blue,1,C]"


def test_solver_pet_shop_sample():
    # Official exam binary sample (cat/dog)
    q = (
        "题目: 每个宠物店有5只宠物;有猫、狗两种宠物店;"
        "小爱领养2只猫,5只狗;小宠、小动都领养2只猫;"
        "小乐领养3只猫,5只狗;小欢领养4只狗。"
        "设猫宠物店数量为x,狗宠物店数量为y,计算x、y分别为多少"
    )
    # cats: 2+2*2+3 = 9 → ceil(9/5)=2; dogs: 5+5+4=14 → ceil(14/5)=3
    x, y = solve_xy_from_question(q)
    assert (x, y) == (2, 3)
    cmd = question_to_command(q)
    assert cmd == "[red,2,C;blue,3,A]"


def test_solver_dou_lingyang_multiply():
    q = (
        "题目: 每个宠物店有5只宠物;有猫、狗两种宠物店;"
        "小A、小B、小C都领养4只猫;小D领养1只狗。"
        "设猫宠物店数量为x,狗宠物店数量为y,计算x、y分别为多少"
    )
    assert solve_xy_from_question(q) == (3, 1)
    assert question_to_command(q) == "[red,3,A;blue,1,C]"


def test_extract_xy_json_variants():
    assert _extract_xy_json('{"x":4,"y":1}') == (4, 1)
    assert _extract_xy_json('答案是 {"x": 2, "y": 3} 完成') == (2, 3)


def test_llm_solver_local_backend():
    q = (
        "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
        "小帅、小丽都需要7件T恤;小娜需要2件T恤,1条裤子;小杰需要1条裤子。"
        "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。"
    )
    cmd, x, y = question_to_command_llm(q, backend="local")
    assert (x, y) == (4, 1)
    assert cmd == "[red,4,A;blue,1,C]"


def test_direct_command_chinese():
    from yzbot_task.local_solver import try_parse_direct_command

    cmd, x, y = try_parse_direct_command("5个红色物块，去A区")
    assert cmd == "[red,5,A]" and (x, y) == (5, 0)
    cmd, x, y = try_parse_direct_command("抓取两个蓝色到C区")
    assert cmd == "[blue,2,C]" and (x, y) == (0, 2)
    cmd, x, y = question_to_command_llm("5个红色物块，去A区", backend="local")
    assert cmd == "[red,5,A]" and (x, y) == (5, 0)
    # Official word problems still use ceil + zone map
    assert try_parse_direct_command(
        "题目: 每个衣柜有5件衣物;小娜需要2件T恤,1条裤子。"
    ) is None


def test_official_xy_and_zone_threshold():
    from yzbot_task.local_solver import is_official_xy, map_xy_to_tasks

    assert is_official_xy(4, 1) and is_official_xy(2, 3)
    assert not is_official_xy(5, 0) and not is_official_xy(3, 3)
    # 规则例：≥3→A，否则 C（不是 >0 / >1）
    tasks = map_xy_to_tasks(4, 1)
    assert str(tasks[0]) == "red,4,A" and str(tasks[1]) == "blue,1,C"
    tasks = map_xy_to_tasks(2, 3)
    assert str(tasks[0]) == "red,2,C" and str(tasks[1]) == "blue,3,A"
    tasks = map_xy_to_tasks(1, 1)  # 都 <3 → 双 C
    assert str(tasks[0]) == "red,1,C" and str(tasks[1]) == "blue,1,C"


def test_format_roundtrip():
    tasks = parse_command("[red,1,A]")
    assert format_command(tasks) == "[red,1,A]"


def test_route_planner_reorders_for_shorter_tour():
    from yzbot_task.route_planner import plan_route

    cmd = "[red,2,A;blue,1,C]"
    plan = plan_route(cmd, start=(0.0, 0.0))
    assert len(plan.legs) == 3
    colors = [c for c, _i, _z in plan.legs]
    assert colors.count("red") == 2 and colors.count("blue") == 1
    assert ("red", "A") in {(c, z) for c, _i, z in plan.legs}
    assert any(c == "blue" and z == "C" for c, _i, z in plan.legs)
    reds = [i for c, i, _z in plan.legs if c == "red"]
    blues = [i for c, i, _z in plan.legs if c == "blue"]
    assert len(reds) == len(set(reds))
    assert len(blues) == len(set(blues))
    assert plan.total_m > 0
    assert "#" in plan.as_command()


def test_route_planner_respects_fixed_ids():
    from yzbot_task.route_planner import plan_route

    plan = plan_route("[red#3,B;blue#1,A]", start=(0.0, 0.0))
    assert plan.legs[0][1] == 3 or plan.legs[1][1] == 3
    ids = {(c, i) for c, i, _z in plan.legs}
    assert ("red", 3) in ids and ("blue", 1) in ids
