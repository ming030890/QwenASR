//! Shared compact trie. Terminal IDs retain the vocabulary's lexical order.
use std::collections::BTreeMap;
const NONE: u32 = u32::MAX;
struct Node { first: u32, count: u32, terminal: u32 }
struct Edge { letter: u8, child: u32 }
pub(crate) struct SpellingIndex { nodes: Vec<Node>, edges: Vec<Edge> }
impl SpellingIndex {
    pub fn new<'a>(words: impl Iterator<Item=&'a str>) -> Self {
        let mut nodes = vec![(BTreeMap::<u8,usize>::new(), NONE)];
        for (id, word) in words.enumerate() {
            let mut node = 0;
            for letter in word.bytes() {
                node = if let Some(&child) = nodes[node].0.get(&letter) { child } else {
                    let child = nodes.len();
                    nodes.push((BTreeMap::new(), NONE));
                    nodes[node].0.insert(letter, child);
                    child
                };
            }
            nodes[node].1 = u32::try_from(id).expect("Spelling vocabulary too large");
        }
        let mut result = Self { nodes: Vec::with_capacity(nodes.len()), edges: Vec::new() };
        for (children, terminal) in nodes {
            result.nodes.push(Node { first: result.edges.len() as u32, count: children.len() as u32, terminal });
            result.edges.extend(children.into_iter().map(|(letter,child)| Edge { letter, child: child as u32 }));
        }
        result
    }
    fn children(&self, node: usize) -> &[Edge] {
        let n = &self.nodes[node];
        &self.edges[n.first as usize..(n.first+n.count) as usize]
    }
    fn prefix(&self, input: &[u8]) -> Option<usize> {
        let mut node = 0;
        for &letter in input {
            let edges = self.children(node);
            node = edges[edges.binary_search_by_key(&letter, |e| e.letter).ok()?].child as usize;
        }
        Some(node)
    }
    fn descendants(&self, node: usize, out: &mut Vec<(usize,u8)>) -> Result<(),String> {
        crate::suggestion_request::check()?;
        let n = &self.nodes[node];
        if n.terminal != NONE { out.push((n.terminal as usize, 0)); }
        for edge in self.children(node) { self.descendants(edge.child as usize, out)?; }
        Ok(())
    }
    // Same OSA recurrence and row pruning as cnn_spelling::distance, shared
    // once per trie prefix instead of recomputing it for every complete word.
    fn fuzzy(&self, node: usize, input: &[u8], radius: usize, depth: usize, last: u8,
        previous: &[u8;33], before_previous: &[u8;33], out: &mut Vec<(usize,u8)>) -> Result<(),String> {
        crate::suggestion_request::check()?;
        for edge in self.children(node) {
            let mut current = [0u8;33];
            current[0] = (depth+1) as u8;
            for j in 1..=input.len() {
                current[j] = (previous[j]+1).min(current[j-1]+1)
                    .min(previous[j-1]+u8::from(edge.letter != input[j-1]));
                if depth > 0 && j > 1 && edge.letter == input[j-2] && last == input[j-1] {
                    current[j] = current[j].min(before_previous[j-2]+1);
                }
            }
            if *current[..=input.len()].iter().min().unwrap() as usize > radius { continue; }
            let terminal = self.nodes[edge.child as usize].terminal;
            if terminal != NONE && current[input.len()] as usize <= radius {
                out.push((terminal as usize,current[input.len()]));
            }
            if depth+1 < 32 {
                self.fuzzy(edge.child as usize,input,radius,depth+1,edge.letter,&current,previous,out)?;
            }
        }
        Ok(())
    }
    pub fn matches(&self, input: &str, radius: usize, completions: bool) -> Result<Vec<(usize,u8)>,String> {
        let mut out = Vec::new();
        if input.is_empty() || input.len() > 32 { return Ok(out); }
        if completions {
            if let Some(node) = self.prefix(input.as_bytes()) { self.descendants(node, &mut out)?; }
        }
        let mut previous = [0u8;33];
        for j in 0..=input.len() { previous[j] = j as u8; }
        self.fuzzy(0,input.as_bytes(),radius,0,0,&previous,&[0u8;33],&mut out)?;
        // Completion distance zero wins over a fuzzy match for the same word.
        out.sort_unstable();
        out.dedup_by_key(|entry| entry.0);
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trie_matches_linear_prefix_and_osa_search() {
        let mut words = vec![String::new()];
        for _ in 0..5 {
            let mut next = Vec::new();
            for word in &words { for c in ['a','b','\''] { next.push(format!("{word}{c}")); } }
            words.extend(next);
            words.sort(); words.dedup();
        }
        words.retain(|w| !w.is_empty());
        words.extend(["x".repeat(32), "x".repeat(31), "transposition".into()]);
        words.sort(); words.dedup();
        let index = SpellingIndex::new(words.iter().map(String::as_str));
        for input in words.iter().chain(["baabx","x","transpositoin"].iter().map(|s| *s).map(str::to_owned).collect::<Vec<_>>().iter()) {
            for radius in 1..=3 { for completions in [false,true] {
                let expected: Vec<_> = words.iter().enumerate().filter_map(|(id,w)| {
                    if completions && w.starts_with(input) { Some((id,0)) } else {
                        crate::cnn_spelling::distance(input.as_bytes(), w.as_bytes(),radius).map(|d| (id,d))
                    }
                }).collect();
                assert_eq!(index.matches(input,radius,completions).unwrap(),expected,"{input} {radius} {completions}");
            } }
        }
    }
}
