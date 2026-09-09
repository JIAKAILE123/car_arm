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
