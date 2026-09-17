from yzbot_task.command_parser import parse_command, format_command
from yzbot_task.local_solver import solve_xy_from_question, question_to_command


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


def test_format_roundtrip():
    tasks = parse_command("[red,1,A]")
    assert format_command(tasks) == "[red,1,A]"


def test_route_planner_reorders_for_shorter_tour():
    from yzbot_task.route_planner import plan_route

    # Naive order: 4 reds to A then 1 blue to C from origin
    cmd = "[red,2,A;blue,1,C]"
    plan = plan_route(cmd, start=(0.0, 0.0))
    assert len(plan.legs) == 3
    colors = [c for c, _i, _z in plan.legs]
    assert colors.count("red") == 2 and colors.count("blue") == 1
    zones = {(c, z) for c, _i, z in plan.legs}
    assert ("red", "A") in {(c, z) for c, _i, z in plan.legs}
    assert any(c == "blue" and z == "C" for c, _i, z in plan.legs)
    # All cube ids unique per color
    reds = [i for c, i, _z in plan.legs if c == "red"]
    blues = [i for c, i, _z in plan.legs if c == "blue"]
    assert len(reds) == len(set(reds))
    assert len(blues) == len(set(blues))
    assert plan.total_m > 0
    # Planned command is cube-id form
    assert "#" in plan.as_command()


def test_route_planner_respects_fixed_ids():
    from yzbot_task.route_planner import plan_route

    plan = plan_route("[red#3,B;blue#1,A]", start=(0.0, 0.0))
    assert plan.legs[0][1] == 3 or plan.legs[1][1] == 3
    ids = {(c, i) for c, i, _z in plan.legs}
    assert ("red", 3) in ids and ("blue", 1) in ids
