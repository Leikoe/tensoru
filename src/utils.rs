pub trait Prod {
    fn prod(&self) -> usize;
}

impl<const N: usize> Prod for [usize; N] {
    fn prod(&self) -> usize {
        self.iter().copied().reduce(|acc, e| acc * e).unwrap_or(0)
    }
}

impl Prod for &[usize] {
    fn prod(&self) -> usize {
        self.iter().copied().reduce(|acc, e| acc * e).unwrap_or(0)
    }
}

impl Prod for Vec<usize> {
    fn prod(&self) -> usize {
        self.iter().copied().reduce(|acc, e| acc * e).unwrap_or(0)
    }
}

// pub fn toposort<T: Eq + std::hash::Hash + Clone + std::fmt::Debug>(
//     graph: &HashMap<T, Vec<T>>,
// ) -> Result<Vec<T>, &'static str> {
//     let mut visited = HashSet::new(); // Tracks all visited nodes.
//     let mut temp_mark = HashSet::new(); // Tracks temporary marks for detecting cycles.
//     let mut result = Vec::new(); // Stores the sorted nodes.

//     // Helper function for DFS
//     fn visit<T: Eq + std::hash::Hash + Clone + std::fmt::Debug>(
//         node: &T,
//         graph: &HashMap<T, Vec<T>>,
//         visited: &mut HashSet<T>,
//         temp_mark: &mut HashSet<T>,
//         result: &mut Vec<T>,
//     ) -> Result<(), &'static str> {
//         if temp_mark.contains(node) {
//             return Err("Graph contains a cycle");
//         }
//         if !visited.contains(node) {
//             temp_mark.insert(node.clone());
//             if let Some(neighbors) = graph.get(node) {
//                 for neighbor in neighbors {
//                     visit(neighbor, graph, visited, temp_mark, result)?;
//                 }
//             }
//             temp_mark.remove(node);
//             visited.insert(node.clone());
//             result.push(node.clone());
//         }
//         Ok(())
//     }

//     // Visit each node in the graph.
//     for node in graph.keys() {
//         visit(node, graph, &mut visited, &mut temp_mark, &mut result)?;
//     }

//     // Reverse the result to get the correct topological order.
//     result.reverse();
//     Ok(result)
// }
