#!/usr/bin/env python3
"""hard-maze-maker — 世界一難しい迷路ジェネレータ.

人間が迷路を解くときの定番戦略を体系的に無効化することを狙って
「難しさ」を最大化した迷路を生成する。

無効化する戦略:
  * 壁伝い法 (右手/左手法) …… braid でループを作り単純連結グラフを崩す
  * 行き止まり潰し法          …… 行き止まりを一部接続してループ化
  * 最短経路の直感            …… グラフ直径の両端を入口/出口にして解を最長化

使い方:
    python3 maze.py --width 41 --height 41 --candidates 40 --seed 42
    python3 maze.py --render ascii
    python3 maze.py --render svg --out maze.svg
"""
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

# セル間の 4 近傍 (dx, dy)
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class Maze:
    """壁を各セルのビットフラグで持つグリッド迷路.

    walls[y][x] の bit: 1=N, 2=S, 4=E, 8=W が「壁がある」ことを表す。
    生成直後は全セルが四方を壁で囲まれた状態から通路を彫っていく。
    """

    width: int
    height: int
    walls: list[list[int]] = field(default_factory=list)
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (0, 0)

    N, S, E, W = 1, 2, 4, 8
    _OPP = {N: S, S: N, E: W, W: E}
    _DELTA = {N: (0, -1), S: (0, 1), E: (1, 0), W: (-1, 0)}

    def __post_init__(self) -> None:
        if not self.walls:
            self.walls = [[15 for _ in range(self.width)] for _ in range(self.height)]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def carve(self, x: int, y: int, direction: int) -> None:
        """(x, y) から direction 方向の壁を取り除いて通路でつなぐ."""
        dx, dy = self._DELTA[direction]
        nx, ny = x + dx, y + dy
        if not self.in_bounds(nx, ny):
            return
        self.walls[y][x] &= ~direction
        self.walls[ny][nx] &= ~self._OPP[direction]

    def linked(self, x: int, y: int, direction: int) -> bool:
        """(x, y) が direction 方向のセルと通路でつながっているか."""
        return not (self.walls[y][x] & direction)

    def neighbors(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        """通路でつながっている隣接セルを返す."""
        for d, (dx, dy) in self._DELTA.items():
            if self.linked(x, y, d):
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    yield nx, ny

    # ------------------------------------------------------------------ 生成

    @classmethod
    def generate(cls, width: int, height: int, rng: random.Random,
                 braid: float = 0.35) -> "Maze":
        """recursive backtracker で完全迷路を掘り、braid でループを加える.

        recursive backtracker は行き止まりが少なく通路が長く蛇行するため、
        解が長くなりやすく「難しい」迷路の土台に向く。
        """
        maze = cls(width, height)
        maze._recursive_backtracker(rng)
        if braid > 0:
            maze._braid(rng, braid)
        maze._place_start_goal()
        return maze

    def _recursive_backtracker(self, rng: random.Random) -> None:
        sx, sy = rng.randrange(self.width), rng.randrange(self.height)
        stack = [(sx, sy)]
        visited = {(sx, sy)}
        while stack:
            x, y = stack[-1]
            unvisited = []
            for d, (dx, dy) in self._DELTA.items():
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny) and (nx, ny) not in visited:
                    unvisited.append((d, nx, ny))
            if not unvisited:
                stack.pop()
                continue
            d, nx, ny = rng.choice(unvisited)
            self.carve(x, y, d)
            visited.add((nx, ny))
            stack.append((nx, ny))

    def _dead_ends(self) -> list[tuple[int, int]]:
        ends = []
        for y in range(self.height):
            for x in range(self.width):
                if sum(1 for _ in self.neighbors(x, y)) == 1:
                    ends.append((x, y))
        return ends

    def _braid(self, rng: random.Random, p: float) -> None:
        """行き止まりを確率 p で近傍とつなぎループを作る.

        ループを作ると壁伝い法・行き止まり潰し法が破綻し、難易度が上がる。
        """
        ends = self._dead_ends()
        rng.shuffle(ends)
        for x, y in ends:
            if sum(1 for _ in self.neighbors(x, y)) != 1:
                continue  # 既に別の braid で解消済み
            if rng.random() > p:
                continue
            # 壁でふさがれている方向のうち、盤面内の隣を優先的につなぐ
            blocked = []
            for d, (dx, dy) in self._DELTA.items():
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny) and not self.linked(x, y, d):
                    # なるべく別の行き止まり以外(=合流)を選ぶと良いループになる
                    deg = sum(1 for _ in self.neighbors(nx, ny))
                    blocked.append((deg, d))
            if blocked:
                blocked.sort(reverse=True)  # 次数の高い隣を優先
                self.carve(x, y, blocked[0][1])

    def _place_start_goal(self) -> None:
        """グラフ直径の両端を入口・出口にして解の最短距離を最長化する."""
        far_from_origin, _ = self._bfs_farthest((0, 0))
        self.start, _ = self._bfs_farthest(far_from_origin)
        self.goal, _ = self._bfs_farthest(self.start)

    # ------------------------------------------------------------- 解析/難易度

    def _bfs(self, src: tuple[int, int]) -> dict[tuple[int, int], int]:
        dist = {src: 0}
        q = deque([src])
        while q:
            x, y = q.popleft()
            for nx, ny in self.neighbors(x, y):
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
        return dist

    def _bfs_farthest(self, src: tuple[int, int]) -> tuple[tuple[int, int], int]:
        dist = self._bfs(src)
        cell = max(dist, key=dist.get)
        return cell, dist[cell]

    def solve(self) -> list[tuple[int, int]]:
        """start から goal への最短経路 (BFS) を返す."""
        prev: dict[tuple[int, int], tuple[int, int]] = {self.start: self.start}
        q = deque([self.start])
        while q:
            cur = q.popleft()
            if cur == self.goal:
                break
            for nb in self.neighbors(*cur):
                if nb not in prev:
                    prev[nb] = cur
                    q.append(nb)
        if self.goal not in prev:
            return []
        path = [self.goal]
        while path[-1] != self.start:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def difficulty(self) -> dict[str, float]:
        """難易度の各指標と総合スコアを算出する.

        - solution_len : 最短解の長さ (長いほど難しい)
        - decisions    : 分岐点 (次数>=3) の数 (迷いやすさ)
        - dead_ends    : 行き止まり数
        - loops        : 独立ループ数 (壁伝い法を破る指標)
        - tortuosity   : 解経路の曲がり回数 / 解長 (くねり具合)
        """
        path = self.solve()
        cells = self.width * self.height
        edges = 0
        decisions = 0
        for y in range(self.height):
            for x in range(self.width):
                deg = sum(1 for _ in self.neighbors(x, y))
                edges += deg
                if deg >= 3:
                    decisions += 1
        edges //= 2
        loops = edges - cells + 1  # 連結グラフの独立閉路数 (循環ランク)

        turns = 0
        for i in range(1, len(path) - 1):
            (ax, ay), (bx, by), (cx, cy) = path[i - 1], path[i], path[i + 1]
            if (bx - ax, by - ay) != (cx - bx, cy - by):
                turns += 1
        tortuosity = turns / max(1, len(path) - 1)

        score = (
            len(path) * 1.0
            + decisions * 0.8
            + loops * 2.0
            + tortuosity * len(path) * 0.5
        )
        return {
            "solution_len": len(path),
            "decisions": decisions,
            "dead_ends": len(self._dead_ends()),
            "loops": loops,
            "tortuosity": round(tortuosity, 3),
            "score": round(score, 1),
        }

    # ---------------------------------------------------------------- 描画

    def to_ascii(self) -> str:
        """2 セル幅の ASCII で描画。S=入口, G=出口。"""
        rows = ["+" + "".join("--+" if self.walls[0][x] & self.N else "  +"
                              for x in range(self.width))]
        # 上辺だけは全周を壁として描く
        rows = ["+" + "--+" * self.width]
        for y in range(self.height):
            top = "|" if self.walls[y][0] & self.W else " "
            bottom = "+"
            for x in range(self.width):
                if (x, y) == self.start:
                    cell = " S"
                elif (x, y) == self.goal:
                    cell = " G"
                else:
                    cell = "  "
                top += cell + ("|" if self.walls[y][x] & self.E else " ")
                bottom += ("--" if self.walls[y][x] & self.S else "  ") + "+"
            rows.append(top)
            rows.append(bottom)
        return "\n".join(rows)

    def to_svg(self, cell: int = 18, path: bool = False) -> str:
        pad = cell
        w = self.width * cell + pad * 2
        h = self.height * cell + pad * 2
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">',
            f'<rect width="{w}" height="{h}" fill="#0b1021"/>',
        ]
        if path:
            pts = " ".join(
                f"{pad + x * cell + cell / 2},{pad + y * cell + cell / 2}"
                for x, y in self.solve()
            )
            lines.append(
                f'<polyline points="{pts}" fill="none" stroke="#38bdf8" '
                f'stroke-width="{cell * 0.32:.1f}" stroke-linejoin="round" '
                f'stroke-linecap="round" opacity="0.55"/>'
            )
        seg = []
        for y in range(self.height):
            for x in range(self.width):
                px, py = pad + x * cell, pad + y * cell
                if self.walls[y][x] & self.N:
                    seg.append((px, py, px + cell, py))
                if self.walls[y][x] & self.S:
                    seg.append((px, py + cell, px + cell, py + cell))
                if self.walls[y][x] & self.E:
                    seg.append((px + cell, py, px + cell, py + cell))
                if self.walls[y][x] & self.W:
                    seg.append((px, py, px, py + cell))
        for x1, y1, x2, y2 in seg:
            lines.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>'
            )
        for (cx, cy), color in ((self.start, "#22c55e"), (self.goal, "#ef4444")):
            lines.append(
                f'<circle cx="{pad + cx * cell + cell / 2}" '
                f'cy="{pad + cy * cell + cell / 2}" r="{cell * 0.3:.1f}" '
                f'fill="{color}"/>'
            )
        lines.append("</svg>")
        return "\n".join(lines)


def make_hardest(width: int, height: int, candidates: int, seed: int | None,
                 braid: float) -> tuple[Maze, dict, list[dict]]:
    """candidates 個生成して難易度スコア最大の迷路を選ぶ."""
    master = random.Random(seed)
    best: Maze | None = None
    best_diff: dict | None = None
    trials: list[dict] = []
    for i in range(candidates):
        rng = random.Random(master.random())
        maze = Maze.generate(width, height, rng, braid=braid)
        diff = maze.difficulty()
        trials.append(diff)
        if best_diff is None or diff["score"] > best_diff["score"]:
            best, best_diff = maze, diff
    assert best is not None and best_diff is not None
    return best, best_diff, trials


def main() -> None:
    p = argparse.ArgumentParser(description="世界一難しい迷路ジェネレータ")
    p.add_argument("--width", type=int, default=31)
    p.add_argument("--height", type=int, default=31)
    p.add_argument("--candidates", type=int, default=30,
                   help="生成して比較する候補数 (多いほど難しい迷路を選抜)")
    p.add_argument("--braid", type=float, default=0.35,
                   help="行き止まりをループ化する割合 0..1")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--render", choices=["ascii", "svg", "none"], default="ascii")
    p.add_argument("--out", type=str, default=None, help="SVG 出力先ファイル")
    p.add_argument("--show-path", action="store_true", help="SVG に解を重ねる")
    args = p.parse_args()

    maze, diff, _ = make_hardest(
        args.width, args.height, args.candidates, args.seed, args.braid
    )

    if args.render == "ascii":
        print(maze.to_ascii())
    elif args.render == "svg":
        svg = maze.to_svg(path=args.show_path)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(svg)
            print(f"wrote {args.out}")
        else:
            print(svg)

    print("difficulty:", json.dumps(diff, ensure_ascii=False))


if __name__ == "__main__":
    main()
