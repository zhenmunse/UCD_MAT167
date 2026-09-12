# MAT 167: PageRank and a Simple Search Engine

A Python implementation of PageRank for a directed network of 12 webpages
(A–L), followed by four fixed Boolean searches over 15 car-brand keywords.
The assignment data and requirements are recorded in
[pagerank_prompt.md](pagerank_prompt.md). The adjacency list in that file is
the authoritative graph input.

## Requirements and Run Instructions

- Python 3
- NumPy (the only external dependency)

From the repository directory, install NumPy and run the script:

```sh
python -m pip install numpy
python pagerank.py
```

No command-line arguments or interactive input are required. All results are
printed to the terminal with English labels, page names, and formatted numbers.
The output can be captured in screenshots for a PDF submission. The script does
not create a PDF or other output files.

## Implementation

All implementation code is in [pagerank.py](pagerank.py), with detailed English
comments explaining the NumPy operations for readers familiar with MATLAB.
The script prints the following nine sections in order:

1. **Raw matrix G:** Build the adjacency matrix and normalize each nonempty row
   by its number of outgoing links. Print G and its row sums.
2. **Dangling-node correction:** Replace page L's zero row with a uniform
   distribution over all 12 pages, including L itself. Print the corrected G
   and its row sums.
3. **Eigendecomposition of Gᵀ:** Select the eigenvector whose eigenvalue is
   closest to 1, normalize it to have L1 norm 1, and print the PageRank ranking.
4. **Power method for G:** Start from a uniform probability vector and iterate
   until the L1 change is at most `1e-10`. Print the iteration count, vector,
   and difference from the eigenvector result.
5. **Teleportation:** Construct `G_tilde = 0.85 * G + 0.15 * E`, where every
   entry of E is `1/12`. Print the matrix and its row sums.
6. **Eigendecomposition of G_tildeᵀ:** Compute the new PageRank vector and
   compare the rankings with and without teleportation side by side.
7. **Power method for G_tilde:** Repeat the convergence calculation and compare
   it with the teleportation eigenvector result.
8. **Term-document matrix T:** Print the binary 15 × 12 matrix, with car brands
   in alphabetical order as rows and webpages A–L as columns.
9. **Search queries:** Evaluate the four fixed queries and rank matching pages
   using the PageRank scores from G_tilde.

Rows of each transition matrix represent source pages; columns represent
destination pages. Accordingly, the power method uses a row-vector update
`previous_vector @ matrix`, while eigendecomposition uses the transpose.
Calculations retain full numerical precision; displayed matrix entries are
rounded to four decimal places.

## Boolean Search

Each keyword receives its own one-hot query vector q. The product `q @ T`
identifies documents containing that keyword. Boolean operations are then
applied to the separate matching vectors:

- **SINGLE:** The keyword is present.
- **AND:** Both keywords are present.
- **OR:** At least one keyword is present.
- **BUT NOT:** The first keyword is present and the second is absent.

The four queries are hardcoded; the script does not parse arbitrary queries.
Matching pages retain their global teleportation PageRank scores without
renormalization within the result set.

| Query | Matching pages (alphabetical) | Ranked results (highest first) |
| --- | --- | --- |
| Nissan | A, C, F, I, L | I > F > L > A > C |
| Mercedes AND Jaguar | A, B, C, L | L > A > B > C |
| Toyota OR Porsche | A, B, C, D, E, F, G, H, I, J, K, L | D > I > G > F > E > L > K > J > A > H > B > C |
| Mazda BUT NOT Volkswagen | A, G, J, K, L | G > L > K > J > A |

## Numerical Verification

The implementation was run and checked for stochastic row sums, nonnegative
normalized PageRank vectors, stationary-equation residuals, agreement between
the eigenvector and power methods, and all four expected query result sets.
The observed power-method results were:

| Matrix | Iterations | L1 difference from eigenvector result |
| --- | ---: | ---: |
| G after dangling-node correction | 35 | approximately 1.75e-11 |
| G_tilde | 28 | approximately 1.70e-11 |

Small floating-point differences may occur across NumPy environments. The
power method has a limit of 100,000 iterations and raises an error if it does
not converge.

## License

This project is available under the [MIT License](LICENSE).
