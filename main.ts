// Sokoban solver for make.fun's sokoban-solver-01: a push-optimal search, room by room.
//
// A level is a hallway with one to four small rooms off it, each holding three or
// four boxes and as many goals, and a box pushed out of a room into the hallway can
// never come back, so the rooms are independent: the level's push optimum is the sum
// of its rooms' optima. This program finds each room, solves it with a breadth-first
// search over pushes (the first solution found is push-optimal), and joins the rooms'
// move strings, walking the player between rooms through the hallway.
//
// A search state is the boxes' squares plus the player's reachable region, named by
// its lowest square: two states that differ only in where the player stands inside
// one region offer the same pushes, so they are one state. Two deadlocks are pruned:
// a box on a square from which no push can ever reach a goal, and a frozen 2x2 block
// of walls and boxes with a box off its goal. The search is bounded by a count of
// states, never by the clock, so two runs answer byte for byte alike.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The most states one room's search keeps before it gives up. */
const MAX_STATES = 250_000;
/** The most states the whole-level fallback search keeps. */
const FALLBACK_STATES = 100_000;

/** [dx, dy, the step's letter, the push's letter], up, down, left, right. */
const DIRS: Array<readonly [number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is a wall whatever a line shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * The squares a box can stand on and still reach a goal: pulled back from the goals.
 * A square y is live when a box there can be pushed to a live square x with the
 * player standing at y - (x - y), the square beyond y away from x.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const size = wall.length;
  const live = new Uint8Array(size);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= size) continue;
      const behind = y - d;
      if (behind < 0 || behind >= size) continue;
      if (wall[y] === 0 && wall[behind] === 0 && live[y] !== 1) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * The squares the player reaches from `from` around walls and the boxes in
 * `occupied`, and the lowest of them: the name of the player's region. The flood
 * marks its squares in `seen` with `stamp` so no array is cleared between calls.
 */
function regionLow(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  const size = wall.length;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= size) continue;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/**
 * Whether the box just pushed to `t` sits in a frozen 2x2 block: four squares of
 * walls and boxes, one of them a box off its goal. Such a block never moves again.
 */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  const size = wall.length;
  const filled = (q: number): boolean => wall[q] === 1 || occupied[q] === 1;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= size) continue;
    if (!filled(a) || !filled(b) || !filled(c) || !filled(d)) continue;
    if ((occupied[a] === 1 && goal[a] !== 1) || (occupied[b] === 1 && goal[b] !== 1) || (occupied[c] === 1 && goal[c] !== 1) || (occupied[d] === 1 && goal[d] !== 1)) return true;
  }
  return false;
}

/** A solved level: how many pushes it took, and the player's moves in LURD. */
interface PushSolution {
  pushes: number;
  moves: string;
}

/** A state's key: its region's lowest square and the boxes' squares, in order. */
function stateKey(boxes: number[], low: number): string {
  return `${low}|${boxes.join(',')}`;
}

/**
 * The push-optimal solution of `level`: a breadth-first search over pushes, so the
 * first solved state it meets is optimal. It keeps at most `maxStates` states, then
 * gives up. Returns null when the level has no solution or the search gives up.
 */
function solvePushes(level: Level, maxStates: number): PushSolution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b] === 1);
  if (level.player < 0) return null;
  if (level.boxes.some((b) => live[b] !== 1)) return null;

  // The states in the order found: their boxes, the player's square after the push
  // that made them, and that push (the box's square before and after it).
  const statesBoxes: number[][] = [level.boxes.slice()];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const index = new Map<string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = regionLow(level, occupied, level.player, seen, ++stamp, stack);
  index.set(stateKey(level.boxes, startLow), 0);
  for (const b of level.boxes) occupied[b] = 0;

  let found = solved(level.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    regionLow(level, occupied, statesPlayer[head]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size) continue;
        // The player must reach the square behind the box, the target must be open
        // floor, and a box pushed there must still be able to reach a goal.
        if (seen[b - d] !== here || wall[t] === 1 || occupied[t] === 1 || live[t] !== 1) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key = '';
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          // The player ends on the box's old square: the new state's region starts there.
          const low = regionLow(level, occupied, b, mark, ++stamp, stack);
          key = stateKey(next, low);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || index.has(key)) continue;
        if (statesBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, statesBoxes.length);
        statesBoxes.push(next!);
        statesPlayer.push(b);
        parent.push(head);
        pushFrom.push(b);
        pushTo.push(t);
        if (solved(next!)) {
          found = statesBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // The pushes in order, then the moves: walk to each push's square, then push.
  const chain: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of chain) {
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    const at = boxes.indexOf(from);
    boxes[at] = to;
    boxes.sort((a, b) => a - b);
  }
  return { pushes: chain.length, moves };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  const size = wall.length;
  if (from === to) return '';
  const prev = new Int32Array(size).fill(-1);
  const how = new Array<string>(size);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= size) continue;
      if (wall[q] === 1 || occupied[q] === 1 || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, the case aside) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

/** One room: its squares, its boxes and goals, and whether it is solved already. */
interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the hallway row's dead squares out of the floor (a box on
 * the player's row reaches no goal, so the hallway never joins two rooms), and each
 * part left that holds a box is a room. Null when the level is no set of rooms: a box
 * starts in the hallway, a goal lies in a cut square, or some part holds more or
 * fewer boxes than goals.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (wall[i] === 0 && live[i] !== 1) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b] === 1)) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] === 1 || cut[s] === 1 || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (q < 0 || q >= wall.length) continue;
        if (wall[q] === 0 && cut[q] === 0 && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    const r = part[b]!;
    if (r < 0) return null;
    rooms[r]!.boxes.push(b);
  }
  for (const g of goals) {
    const r = part[g]!;
    if (r < 0) return null;
    rooms[r]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn, the
 * other rooms' boxes standing as walls (on their goals once their room is done), the
 * player walking on from where the last room left it. A box never leaves its room, so
 * the rooms' optima add up to the level's. Null when a room's search gives up, or
 * the level is no set of rooms and the search over the whole level gives up.
 */
function solveByRooms(level: Level, maxStates: number): PushSolution | null {
  const rooms = roomsOf(level);
  if (rooms === null) return solvePushes(level, maxStates);
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushes(sub, maxStates);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/** The player's moves for one level: room by room, or the whole level as a fallback. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const byRooms = solveByRooms(level, MAX_STATES);
  if (byRooms !== null) return byRooms.moves;
  const whole = solvePushes(level, FALLBACK_STATES);
  if (whole !== null) return whole.moves;
  return '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
