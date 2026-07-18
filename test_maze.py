"""maze.py の健全性テスト.

    python3 -m pytest test_maze.py      # pytest がある場合
    python3 test_maze.py                # 単体実行 (assert ベース)
"""
import random

from maze import Maze, make_hardest


def _connected(maze: Maze) -> bool:
    """全セルが到達可能か (迷路として連結か)."""
    return len(maze._bfs((0, 0))) == maze.width * maze.height


def test_perfect_maze_is_connected_and_loopless():
    m = Maze.generate(21, 21, random.Random(1), braid=0.0)
    assert _connected(m)
    d = m.difficulty()
    assert d["loops"] == 0  # braid なし = 完全迷路 = ループ0
    assert d["solution_len"] > 0


def test_braided_maze_has_loops_and_stays_connected():
    m = Maze.generate(21, 21, random.Random(1), braid=0.8)
    assert _connected(m)
    assert m.difficulty()["loops"] > 0  # braid あり = ループが生まれる


def test_start_goal_is_solvable_and_distinct():
    m = Maze.generate(15, 15, random.Random(3), braid=0.4)
    assert m.start != m.goal
    path = m.solve()
    assert path and path[0] == m.start and path[-1] == m.goal


def test_make_hardest_picks_highest_score():
    m, diff, trials = make_hardest(15, 15, candidates=15, seed=7, braid=0.4)
    assert diff["score"] == max(t["score"] for t in trials)
    assert _connected(m)


def test_reproducible_with_seed():
    a, da, _ = make_hardest(15, 15, candidates=10, seed=99, braid=0.4)
    b, db, _ = make_hardest(15, 15, candidates=10, seed=99, braid=0.4)
    assert da == db
    assert a.walls == b.walls
    assert a.start == b.start and a.goal == b.goal


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
