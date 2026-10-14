//! English-only port of wmkeyboard FuzzyBeamSearch and TouchModel.
//! Reference: 74bf16e27ff72403bcd55eb942894d2158e9bd2b (MIT, Wasi Master).
//! See data/english/LICENSE. No runtime word-list parsing or ASR dependency.
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

const MAX_POPS: usize = 4096;
const K: usize = 8;
const MAX_EDIT_COST: f64 = 2.0;
const SECOND_EDIT_SURCHARGE: f64 = 2.0;
const CHEAP: f64 = 0.10536051565782628; // -ln(.9)
const MEDIUM: f64 = 0.35667494393873245; // -ln(.7)
const INSERT_FAR: f64 = 1.3862943611198906; // -ln(.25)
const SUB_FAR: f64 = 1.6094379124341003; // -ln(.2)
const EPS: f64 = 1e-9;
// SuggestionEngine's synthetic runner-up keeps an unopposed rare word from firing.
const SOLO_RUNNER_UP_SCORE: f64 = 1.0;
const ROWS: [&[u8]; 3] = [b"qwertyuiop", b"asdfghjkl", b"zxcvbnm"];

#[derive(Clone, Copy, Debug)]
struct Node {
    first: u32,
    count: u32,
    frequency: u32,
    maximum: u32,
    parent: u32,
    label: u8,
}
#[derive(Clone, Copy, Debug)]
struct Edge {
    label: u8,
    child: u32,
}
pub struct Dictionary {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl Dictionary {
    /// EAC1, node/edge counts, 16-byte nodes and 8-byte sorted edges (little endian).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let read = |at: usize| -> Result<u32, String> {
            bytes
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .ok_or_else(|| "Truncated English dictionary".into())
        };
        if bytes.get(..4) != Some(b"EAC1") {
            return Err("Invalid English dictionary version".into());
        }
        let n = read(4)? as usize;
        let e = read(8)? as usize;
        if n == 0 || n > 1_000_000 || e != n - 1 || bytes.len() != 12 + n * 16 + e * 8 {
            return Err("Invalid English dictionary size".into());
        }
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let at = 12 + i * 16;
            let node = Node {
                first: read(at)?,
                count: read(at + 4)?,
                frequency: read(at + 8)?,
                maximum: read(at + 12)?,
                parent: 0,
                label: 0,
            };
            if node.first as usize + node.count as usize > e || node.maximum < node.frequency {
                return Err("Invalid English node".into());
            }
            nodes.push(node);
        }
        let mut edges = Vec::with_capacity(e);
        for i in 0..e {
            let at = 12 + n * 16 + i * 8;
            let label = read(at)?;
            let child = read(at + 4)?;
            if !(label == 39 || (97..=122).contains(&label)) || child == 0 || child as usize >= n {
                return Err("Invalid English edge".into());
            }
            edges.push(Edge {
                label: label as u8,
                child,
            });
        }
        let mut seen = vec![false; n];
        let mut depths = vec![0u8; n];
        seen[0] = true;
        let mut next_edge = 0;
        for i in 0..n {
            let node = nodes[i];
            if node.first as usize != next_edge {
                return Err("Overlapping English edges".into());
            }
            let mut previous = 0;
            for edge in &edges[node.first as usize..(node.first + node.count) as usize] {
                let child = edge.child as usize;
                if child <= i
                    || seen[child]
                    || edge.label <= previous
                    || node.maximum < nodes[child].maximum
                {
                    return Err("Invalid English trie topology".into());
                }
                previous = edge.label;
                if depths[i] >= 64 {
                    return Err("English word exceeds length limit".into());
                }
                depths[child] = depths[i] + 1;
                seen[child] = true;
                nodes[child].parent = i as u32;
                nodes[child].label = edge.label;
            }
            next_edge += node.count as usize;
        }
        if seen.iter().any(|v| !v) {
            return Err("Unreachable English node".into());
        }
        Ok(Self { nodes, edges })
    }
    fn children(&self, node: u32) -> &[Edge] {
        let n = self.nodes[node as usize];
        &self.edges[n.first as usize..(n.first + n.count) as usize]
    }
    fn child(&self, node: u32, label: u8) -> Option<u32> {
        let edges = self.children(node);
        edges
            .binary_search_by_key(&label, |e| e.label)
            .ok()
            .map(|i| edges[i].child)
    }
    fn find(&self, word: &[u8]) -> Option<u32> {
        let mut node = 0;
        for &c in word {
            node = self.child(node, c)?;
        }
        Some(node)
    }
    fn word(&self, mut node: u32) -> String {
        let mut bytes = Vec::with_capacity(24);
        while node != 0 {
            let n = self.nodes[node as usize];
            bytes.push(n.label);
            node = n.parent;
        }
        bytes.reverse();
        String::from_utf8(bytes).unwrap()
    }
    pub fn memory_bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<Node>()
            + self.edges.capacity() * std::mem::size_of::<Edge>()
    }
}

/// Deterministic converter for the pinned upstream `word frequency` source.
pub fn pack(source: &str) -> Result<Vec<u8>, String> {
    #[derive(Default)]
    struct BuildNode {
        edges: BTreeMap<u8, usize>,
        frequency: u32,
        maximum: u32,
    }
    let mut nodes = vec![BuildNode::default()];
    for (line_no, line) in source.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 2
            || fields[0].len() > 64
            || !fields[0]
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c == b'\'')
        {
            return Err(format!("Invalid English word at line {}", line_no + 1));
        }
        let frequency: u32 = fields[1]
            .parse()
            .map_err(|_| format!("Invalid frequency at line {}", line_no + 1))?;
        if frequency == 0 {
            return Err("Zero frequency".into());
        }
        let mut node = 0;
        nodes[0].maximum = nodes[0].maximum.max(frequency);
        for c in fields[0].bytes() {
            let next = if let Some(&next) = nodes[node].edges.get(&c) {
                next
            } else {
                let next = nodes.len();
                nodes.push(BuildNode::default());
                nodes[node].edges.insert(c, next);
                next
            };
            node = next;
            nodes[node].maximum = nodes[node].maximum.max(frequency);
        }
        nodes[node].frequency = nodes[node].frequency.max(frequency);
    }
    let mut bytes = b"EAC1".to_vec();
    let append = |bytes: &mut Vec<u8>, n: u32| bytes.extend_from_slice(&n.to_le_bytes());
    append(&mut bytes, nodes.len() as u32);
    append(&mut bytes, (nodes.len() - 1) as u32);
    let mut first = 0;
    for node in &nodes {
        for value in [first, node.edges.len() as u32, node.frequency, node.maximum] {
            append(&mut bytes, value);
        }
        first += node.edges.len() as u32;
    }
    for node in &nodes {
        for (&label, &child) in &node.edges {
            append(&mut bytes, label as u32);
            append(&mut bytes, child as u32);
        }
    }
    Dictionary::from_bytes(&bytes)?;
    Ok(bytes)
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}
/// Coordinates and scatter are in key-width units, exactly as TouchModel.kt.
pub struct TouchModel {
    pub centers: [Option<Point>; 26],
    pub points: Vec<Option<Point>>,
}
impl Default for TouchModel {
    fn default() -> Self {
        Self {
            centers: [None; 26],
            points: Vec::new(),
        }
    }
}
impl TouchModel {
    fn center(&self, c: u8) -> Option<Point> {
        if c.is_ascii_lowercase() {
            self.centers[(c - b'a') as usize]
        } else {
            None
        }
    }
    fn likelihood(&self, p: Point, c: u8) -> Option<f64> {
        let center = self.center(c)?;
        Some(-((p.x - center.x).powi(2) + (p.y - center.y).powi(2)) / (2.0 * 0.5 * 0.5))
    }
    fn match_cost(&self, pos: usize, c: u8) -> f64 {
        let Some(p) = self.points.get(pos).copied().flatten() else {
            return 0.0;
        };
        let Some(actual) = self.likelihood(p, c) else {
            return 0.0;
        };
        let best = (b'a'..=b'z')
            .filter_map(|key| self.likelihood(p, key))
            .fold(f64::NEG_INFINITY, f64::max);
        (best - actual).clamp(0.0, 1.0)
    }
    fn substitution(&self, pos: usize, expected: u8, label: u8) -> f64 {
        if let Some(p) = self.points.get(pos).copied().flatten() {
            if let (Some(a), Some(b)) = (self.likelihood(p, expected), self.likelihood(p, label)) {
                return CHEAP + (a - b).clamp(0.0, SUB_FAR - CHEAP);
            }
        }
        if adjacent(expected, label) {
            CHEAP
        } else {
            SUB_FAR
        }
    }
}
fn position(c: u8) -> Option<(usize, usize)> {
    ROWS.iter()
        .enumerate()
        .find_map(|(r, row)| row.iter().position(|&key| key == c).map(|i| (r, i)))
}
fn adjacent(a: u8, b: u8) -> bool {
    match (position(a), position(b)) {
        (Some((r, i)), Some((s, j))) => a != b && r.abs_diff(s) <= 1 && i.abs_diff(j) <= 1,
        _ => false,
    }
}
fn same_hand(a: u8, b: u8) -> bool {
    match (position(a), position(b)) {
        (Some((_, i)), Some((_, j))) => (i < 5) == (j < 5),
        _ => false,
    }
}

#[derive(Clone, Copy)]
struct State {
    node: u32,
    pos: usize,
    cost: f64,
    spend: f64,
    edits: u8,
    completed: u8,
}
#[derive(Clone, Copy)]
struct HeapEntry {
    bound: f64,
    id: usize,
}
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.bound == other.bound
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bound
            .total_cmp(&other.bound)
            .then_with(|| other.id.cmp(&self.id))
    }
}
#[derive(Clone, Copy)]
struct Found {
    state: State,
    score: f64,
}
#[derive(Default)]
pub struct Workspace {
    states: Vec<State>,
    heap: BinaryHeap<HeapEntry>,
    found: Vec<Found>,
    pub pops: usize,
}
impl Workspace {
    pub fn memory_bytes(&self) -> usize {
        self.states.capacity() * std::mem::size_of::<State>()
            + self.heap.capacity() * std::mem::size_of::<HeapEntry>()
            + self.found.capacity() * std::mem::size_of::<Found>()
    }
    fn push(&mut self, dict: &Dictionary, state: State, floor: f64) {
        let bound = (1.0 + dict.nodes[state.node as usize].maximum as f64).ln() - state.cost;
        if bound < floor - EPS {
            return;
        }
        let id = self.states.len();
        self.states.push(state);
        self.heap.push(HeapEntry { bound, id });
    }
}
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub word: String,
    pub score: f64,
    pub edit_cost: f64,
    pub edits: u8,
    pub completed_chars: u8,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Prediction {
    pub candidates: Vec<Candidate>,
    pub correction: Option<String>,
    pub plausible_prefix: bool,
    pub known: bool,
}

pub fn search(
    dict: &Dictionary,
    typed: &str,
    touch: &TouchModel,
    ws: &mut Workspace,
) -> Prediction {
    search_with_limits(dict, typed, touch, ws, if typed.len() >= 5 { 2 } else { 1 }, MAX_EDIT_COST)
}

/// Broader spelling alternatives are selectable even when automatic correction is unsafe.
/// Keep the pinned conservative search/confidence policy for automatic replacement.
pub fn suggestions(dict: &Dictionary, typed: &str, touch: &TouchModel, ws: &mut Workspace) -> Prediction {
    let mut result = search(dict, typed, touch, ws);
    if typed.len() < 5 || result.known || result.correction.is_some() { return result; }
    let broad = search_with_limits(dict, typed, touch, ws, 4, 4.5);
    for candidate in broad.candidates {
        if !result.candidates.iter().any(|c| c.word == candidate.word) {
            result.candidates.push(candidate);
        }
    }
    // Several adjacent mistaps can exceed the touch budget even when the
    // spelling is recoverable. Offer geometry-free recovery for explicit taps.
    if !touch.points.is_empty() && result.candidates.iter().filter(|c| c.completed_chars == 0).count() < 3 {
        let recovery = search_with_limits(dict, typed, &TouchModel::default(), ws, 4, 4.5);
        for candidate in recovery.candidates {
            if !result.candidates.iter().any(|c| c.word == candidate.word) {
                result.candidates.push(candidate);
            }
        }
    }
    result.candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.word.cmp(&b.word)));
    result.candidates.truncate(K);
    // Avoid Chinese phrase auto-conversion while a recoverable English word is being typed.
    result.plausible_prefix |= result.candidates.iter().any(|c| c.completed_chars == 0);
    result
}

fn search_with_limits(
    dict: &Dictionary, typed: &str, touch: &TouchModel, ws: &mut Workspace,
    max_edits: u8, max_edit_cost: f64,
) -> Prediction {
    ws.states.clear();
    ws.heap.clear();
    ws.found.clear();
    ws.pops = 0;
    let empty = || Prediction {
        candidates: vec![],
        correction: None,
        plausible_prefix: false,
        known: false,
    };
    if typed.is_empty()
        || typed.len() > 64
        || !typed.bytes().all(|c| c.is_ascii_lowercase() || c == b'\'')
    {
        return empty();
    }
    let typed = typed.as_bytes();
    let mut floor = f64::NEG_INFINITY;
    ws.push(
        dict,
        State {
            node: 0,
            pos: 0,
            cost: 0.0,
            spend: 0.0,
            edits: 0,
            completed: 0,
        },
        floor,
    );
    while let Some(entry) = ws.heap.pop() {
        if ws.pops >= MAX_POPS || entry.bound < floor - EPS {
            break;
        }
        ws.pops += 1;
        let s = ws.states[entry.id];
        if s.pos == typed.len() {
            let frequency = dict.nodes[s.node as usize].frequency;
            if frequency > 0 {
                let score = (1.0 + frequency as f64).ln() - s.cost;
                if let Some(found) = ws.found.iter_mut().find(|c| c.state.node == s.node) {
                    if score > found.score {
                        *found = Found { state: s, score };
                    }
                } else {
                    ws.found.push(Found { state: s, score });
                }
                if ws.found.len() >= K {
                    ws.found
                        .sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
                    floor = ws.found[K - 1].score;
                    // Retain tied scores for deterministic alphabetical selection.
                    ws.found.retain(|c| c.score >= floor - EPS);
                }
            }
            for e in dict.children(s.node) {
                ws.push(
                    dict,
                    State {
                        node: e.child,
                        cost: s.cost + if s.edits > 0 { INSERT_FAR } else { 0.0 },
                        completed: s.completed + 1,
                        ..s
                    },
                    floor,
                );
            }
            continue;
        }
        let expected = typed[s.pos];
        if let Some(child) = dict.child(s.node, expected) {
            ws.push(
                dict,
                State {
                    node: child,
                    pos: s.pos + 1,
                    cost: s.cost + touch.match_cost(s.pos, expected),
                    ..s
                },
                floor,
            );
        }
        if s.edits >= max_edits || s.spend >= max_edit_cost {
            continue;
        }
        let surcharge = if s.edits + 1 >= 2 {
            SECOND_EDIT_SURCHARGE
        } else {
            0.0
        };
        let edit = |node, pos, base, ws: &mut Workspace| {
            if s.spend + base <= max_edit_cost {
                ws.push(
                    dict,
                    State {
                        node,
                        pos,
                        cost: s.cost + base + surcharge,
                        spend: s.spend + base,
                        edits: s.edits + 1,
                        ..s
                    },
                    floor,
                );
            }
        };
        let doubled = (s.pos + 1 < typed.len() && typed[s.pos + 1] == expected)
            || (s.pos > 0 && typed[s.pos - 1] == expected);
        edit(s.node, s.pos + 1, if doubled { CHEAP } else { MEDIUM }, ws);
        for e in dict.children(s.node) {
            if e.label != expected {
                edit(
                    e.child,
                    s.pos + 1,
                    touch.substitution(s.pos, expected, e.label),
                    ws,
                );
            }
            let close = adjacent(expected, e.label)
                || (s.pos > 0
                    && (typed[s.pos - 1] == e.label || adjacent(typed[s.pos - 1], e.label)));
            edit(e.child, s.pos, if close { MEDIUM } else { INSERT_FAR }, ws);
        }
        if s.pos + 1 < typed.len() && expected != typed[s.pos + 1] {
            if let Some(first) = dict.child(s.node, typed[s.pos + 1]) {
                if let Some(second) = dict.child(first, expected) {
                    edit(
                        second,
                        s.pos + 2,
                        if same_hand(expected, typed[s.pos + 1]) {
                            MEDIUM
                        } else {
                            CHEAP
                        },
                        ws,
                    );
                }
            }
        }
    }
    let mut candidates: Vec<_> = ws
        .found
        .iter()
        .map(|c| Candidate {
            word: dict.word(c.state.node),
            score: c.score,
            edit_cost: c.state.spend,
            edits: c.state.edits,
            completed_chars: c.state.completed,
        })
        .collect();
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.word.cmp(&b.word))
    });
    candidates.truncate(K);
    let node = dict.find(typed);
    let known = node
        .map(|i| dict.nodes[i as usize].frequency > 0)
        .unwrap_or(false);
    let correction = choose_correction(&candidates, known, typed.len());
    Prediction {
        plausible_prefix: typed.len() >= 3 && (node.is_some() || correction.is_some()),
        candidates,
        correction,
        known,
    }
}

fn choose_correction(candidates: &[Candidate], known: bool, length: usize) -> Option<String> {
    if known || length < 3 { return None; }
    // WM SuggestionEngine's ASCII QWERTY correction shapes. Multi-edit and
    // edited-then-completed words remain selectable, but never gate a fix.
    let mut fixes: Vec<_> = candidates.iter().take(K).filter_map(|c| {
        let score = match (c.edits, c.completed_chars) {
            (1, 0) => c.score,
            // A missing final letter is an insertion, even though the trie
            // represents it as a zero-cost clean-prefix completion.
            (0, 1) => c.score - MEDIUM,
            _ => return None,
        };
        Some((c, score))
    }).collect();
    fixes.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.word.cmp(&b.0.word)));
    let (top, score) = fixes.first()?;
    let runner = fixes.get(1).map(|c| c.1).unwrap_or(SOLO_RUNNER_UP_SCORE);
    (score - runner >= 4.0_f64.ln()).then(|| top.word.clone())
}

pub fn rerank(prediction: &mut Prediction, scores: &[f32], length: usize) {
    if scores.len() != prediction.candidates.len() || scores.iter().any(|s| !s.is_finite()) {
        return;
    }
    for (candidate, score) in prediction.candidates.iter_mut().zip(scores) {
        candidate.score += *score as f64;
    }
    prediction.candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.word.cmp(&b.word))
    });
    prediction.correction = choose_correction(&prediction.candidates, prediction.known, length);
    prediction.plausible_prefix |= prediction.correction.is_some() && length >= 3;
}

#[cfg(feature = "android")]
mod jni;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_kotlin_reference_parity() {
        #[derive(serde::Deserialize)]
        struct Case {
            word: String,
            touch: bool,
            candidates: Vec<ReferenceCandidate>,
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct ReferenceCandidate {
            word: String,
            score: f64,
            edit_cost: f64,
            edits: u8,
            completed_chars: u8,
        }
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../../data/english/reference-cases.json")).unwrap();
        let dictionary = dict(include_str!("../../data/english/en.txt"));
        for case in cases {
            let mut touch = TouchModel::default();
            if case.touch {
                for (r, row) in ROWS.iter().enumerate() {
                    for (i, &c) in row.iter().enumerate() {
                        touch.centers[(c - b'a') as usize] = Some(Point {
                            x: (i as f32 + [0f32, 0.5, 1.5][r]) as f64,
                            y: r as f64,
                        });
                    }
                }
                touch.points = case
                    .word
                    .bytes()
                    .map(|c| {
                        touch.center(c).map(|p| Point {
                            x: (p.x as f32 + 0.45f32) as f64,
                            y: (p.y as f32 + 0.15f32) as f64,
                        })
                    })
                    .collect();
            }
            let actual = search(&dictionary, &case.word, &touch, &mut Workspace::default());
            assert_eq!(
                actual.candidates.len(),
                case.candidates.len(),
                "{} touch={}",
                case.word,
                case.touch
            );
            for (a, b) in actual.candidates.iter().zip(&case.candidates) {
                assert_eq!(a.word, b.word, "{} touch={}", case.word, case.touch);
                assert!(
                    (a.score - b.score).abs() < 1e-8,
                    "{} {} score {} != {}",
                    case.word,
                    a.word,
                    a.score,
                    b.score
                );
                assert!((a.edit_cost - b.edit_cost).abs() < 1e-8);
                assert_eq!(a.edits, b.edits);
                assert_eq!(a.completed_chars, b.completed_chars);
            }
        }
    }

    #[test]
    fn context_disambiguates_without_overriding_known_words_or_edit_gate() {
        let d = dict("cat 100\ncar 100\n");
        let mut p = search(&d, "caf", &TouchModel::default(), &mut Workspace::default());
        assert!(p.correction.is_none());
        let scores: Vec<_> = p
            .candidates
            .iter()
            .map(|c| if c.word == "cat" { 0. } else { -2. })
            .collect();
        rerank(&mut p, &scores, 3);
        assert_eq!(p.correction.as_deref(), Some("cat"));
        let mut known = search(&d, "car", &TouchModel::default(), &mut Workspace::default());
        let scores = vec![0.; known.candidates.len()];
        rerank(&mut known, &scores, 3);
        assert!(known.correction.is_none());
    }
    fn dict(text: &str) -> Dictionary {
        Dictionary::from_bytes(&pack(text).unwrap()).unwrap()
    }
    fn centre_touches(typed: &str) -> TouchModel {
        let mut touch = TouchModel::default();
        for (row, letters) in ROWS.iter().enumerate() {
            for (column, &letter) in letters.iter().enumerate() {
                touch.centers[(letter - b'a') as usize] = Some(Point {
                    x: column as f64 + [0., 0.5, 1.5][row], y: row as f64,
                });
            }
        }
        touch.points = typed.bytes().map(|letter| touch.center(letter)).collect();
        touch
    }
    #[test]
    fn reported_centre_mistaps_correct_but_multiple_edits_are_selectable_only() {
        let d = dict(include_str!("../../data/english/en.txt"));
        let mut ws = Workspace::default();
        for (typed, expected) in [("engkish", "english"), ("shoukd", "should"),
            ("chineae", "chinese"), ("candjdates", "candidates"), ("kehboard", "keyboard")] {
            let result = suggestions(&d, typed, &centre_touches(typed), &mut ws);
            assert_eq!(result.correction.as_deref(), Some(expected), "{typed}: {result:?}");
        }
        let result = suggestions(&d, "ebgkisb", &centre_touches("ebgkisb"), &mut ws);
        assert!(result.candidates.iter().any(|c| c.word == "english"));
        assert!(result.correction.is_none());
        for typed in ["english", "keyboard", "iPhone", "a1b"] {
            assert!(suggestions(&d, typed, &centre_touches(typed), &mut ws).correction.is_none());
        }
        let ambiguous = dict("cat 100\ncar 100\n");
        assert!(suggestions(&ambiguous, "caf", &centre_touches("caf"), &mut ws).correction.is_none());
    }
    #[test]
    fn wm_replacement_shapes_and_default_confidence() {
        let candidate = |word: &str, score, edits, completed_chars| Candidate {
            word: word.into(), score, edit_cost: SUB_FAR, edits, completed_chars,
        };
        // Two edits neither trigger replacement nor block a one-edit fix.
        let two = candidate("hello", 12., 2, 0);
        let one = candidate("world", 5., 1, 0);
        assert_eq!(choose_correction(&[two.clone(), one.clone()], false, 5).as_deref(), Some("world"));
        assert!(choose_correction(&[two], false, 5).is_none());
        // Edited completions and longer clean completions are suggestions only.
        for c in [candidate("worlds", 12., 1, 1), candidate("worlds", 12., 0, 2)] {
            assert!(choose_correction(&[c.clone()], false, 5).is_none());
            assert_eq!(choose_correction(&[c, one.clone()], false, 5).as_deref(), Some("world"));
        }
        let tail = candidate("keyboard", 5., 0, 1);
        assert_eq!(choose_correction(&[tail.clone()], false, 7).as_deref(), Some("keyboard"));
        // Charge the tail insertion before ranking and comparing confidence.
        assert!(choose_correction(&[tail, candidate("keyboards", 3.5, 1, 0)], false, 7).is_none());
        assert!(choose_correction(&[one.clone(), candidate("words", 4., 1, 0)], false, 5).is_none());
        assert_eq!(choose_correction(&[one.clone(), candidate("words", 3.5, 1, 0)], false, 5).as_deref(), Some("world"));
        assert!(choose_correction(&[one], true, 5).is_none());
    }
    #[test]
    fn closer_one_edit_runner_must_not_hide_behind_a_discounted_two_edit_candidate() {
        let d = dict(include_str!("../../data/english/en.txt"));
        let result = search(&d, "tyoing", &TouchModel::default(), &mut Workspace::default());
        assert!(result.candidates.iter().any(|c| c.word == "typing" && c.edits == 1));
        assert!(result.correction.is_none(), "Ambiguous typing/tying must not silently select tying: {result:?}");
    }
    #[test]
    fn operations_and_conservative_gate() {
        let d = dict("the 10000\nhello 800\napple 2500\nworld 2380\nkeyboard 120\n");
        for (typed, want) in [
            ("teh", "the"),
            ("helllo", "hello"),
            ("aple", "apple"),
            ("wprld", "world"),
        ] {
            let p = search(&d, typed, &TouchModel::default(), &mut Workspace::default());
            assert_eq!(p.correction.as_deref(), Some(want), "{typed}: {p:?}");
        }
        for typed in ["the", "he", "key", "iPhone", "a1b"] {
            assert!(
                search(&d, typed, &TouchModel::default(), &mut Workspace::default())
                    .correction
                    .is_none()
            );
        }
        assert!(
            search(&d, "key", &TouchModel::default(), &mut Workspace::default()).plausible_prefix
        );
    }
    #[test]
    fn ambiguity_and_two_edits_are_suggestions_only() {
        let d = dict("cat 100\ncar 100\nhello 800\n");
        let p = search(&d, "caf", &TouchModel::default(), &mut Workspace::default());
        assert!(p.correction.is_none());
        let rare = dict("abc 1\n");
        let p = search(
            &rare,
            "acb",
            &TouchModel::default(),
            &mut Workspace::default(),
        );
        assert_eq!(p.candidates[0].word, "abc");
        assert!(p.correction.is_none());
        let p = search(
            &d,
            "helolx",
            &TouchModel::default(),
            &mut Workspace::default(),
        );
        assert!(p.correction.is_none());
    }
    #[test]
    fn touch_matches_reference_gaussian() {
        let mut t = TouchModel::default();
        t.centers[0] = Some(Point { x: 0.0, y: 0.0 });
        t.centers[1] = Some(Point { x: 1.0, y: 0.0 });
        t.points.push(Some(Point { x: 0.9, y: 0.0 }));
        assert!((t.likelihood(t.points[0].unwrap(), b'a').unwrap() + 1.62).abs() < 1e-9);
        assert_eq!(t.match_cost(0, b'a'), 1.0);
        assert_eq!(t.match_cost(0, b'b'), 0.0);
        assert!((t.substitution(0, b'a', b'b') - CHEAP).abs() < 1e-9);
        assert_eq!(t.substitution(1, b'q', b'w'), CHEAP);
    }
    #[test]
    fn binary_validation_and_determinism() {
        let a = pack("the 10000\nhello 800\n").unwrap();
        assert_eq!(a, pack("the 10000\nhello 800\n").unwrap());
        for n in 0..a.len() {
            assert!(Dictionary::from_bytes(&a[..n]).is_err());
        }
        assert!(pack("bad_word 1").is_err());
        assert!(pack("word nope").is_err());
        let mut broken = a.clone();
        broken[12..16].copy_from_slice(&1u32.to_le_bytes());
        assert!(Dictionary::from_bytes(&broken).is_err());
    }
    #[test]
    fn full_dictionary_mobile_budget() {
        let bytes = pack(include_str!("../../data/english/en.txt")).unwrap();
        let d = Dictionary::from_bytes(&bytes).unwrap();
        let mut ws = Workspace::default();
        for input in ["teh", "helllo", "keyboadr", "abcdefghijk", "don't", "wprld"] {
            let p = search(&d, input, &TouchModel::default(), &mut ws);
            assert!(ws.pops <= MAX_POPS);
            assert!(p.candidates.len() <= K);
            assert_eq!(
                p.candidates.iter().map(|c| &c.word).collect::<Vec<_>>(),
                search(&d, input, &TouchModel::default(), &mut Workspace::default())
                    .candidates
                    .iter()
                    .map(|c| &c.word)
                    .collect::<Vec<_>>()
            );
        }
        eprintln!(
            "English packed={} resident={} workspace states={}",
            bytes.len(),
            d.memory_bytes(),
            ws.states.capacity()
        );
    }
}
