"""MAT 167: PageRank and four Boolean searches on webpages A through L.

Run with ``python pagerank.py``. NumPy is the only external dependency.
Matrices use source pages as rows and destination pages as columns. Thus
probability row vectors evolve by right multiplication, while the equivalent
column-vector eigenvalue equation uses the transpose of the matrix.
"""

import numpy as np


def print_banner(section_number, title):
    """Separate the nine stages so terminal output is easy to capture."""
    print(f"\n=== Section {section_number}: {title} ===")


def print_matrix(matrix, row_labels, column_labels, integer_entries=False):
    """Print explicit labels and formatted entries instead of raw arrays.

    The wider row-label field accommodates multiword brands in the term matrix.
    NumPy indexing [row, column] corresponds to MATLAB indexing (row, column),
    except that Python indices begin at zero.
    """
    label_width = max(len(str(label)) for label in row_labels)
    print(" " * (label_width + 1) + "".join(f"{label:>8}" for label in column_labels))
    for row_index, label in enumerate(row_labels):
        entries = "".join(
            f"{int(value):8d}" if integer_entries else f"{value:8.4f}"
            for value in matrix[row_index]
        )
        print(f"{label:>{label_width}} {entries}")


def print_row_sums(matrix, page_labels):
    """Sum across columns (axis=1), analogous to MATLAB sum(matrix, 2)."""
    print("Row sums: " + ", ".join(
        f"{page}={value:.6f}"
        for page, value in zip(page_labels, matrix.sum(axis=1))
    ))


def print_vector(vector, page_labels):
    """Display one probability per page with enough precision for comparison."""
    for page, probability in zip(page_labels, vector):
        print(f"  {page}: {probability:.10f}")
    print(f"  L1 norm: {np.linalg.norm(vector, ord=1):.10f}")


def ranking_indices(vector):
    """Sort descending; stable sorting preserves alphabetical order for ties."""
    return np.argsort(-vector, kind="stable")


def print_ranking(vector, page_labels):
    """Print pages from most to least important according to PageRank."""
    print("Ranking: " + " > ".join(page_labels[index] for index in ranking_indices(vector)))


def eigen_pagerank(matrix):
    """Find the stationary column vector from the transpose's eigensystem.

    numpy.linalg.eig returns eigenvectors as columns, as MATLAB eig does.
    Eigenvectors have arbitrary scale and sign. A global sign flip is valid;
    taking entrywise absolute values is not generally an eigenvector operation.
    Only roundoff-sized negative entries are clipped after checking the sign.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    stationary_index = np.argmin(np.abs(eigenvalues - 1.0))
    if abs(eigenvalues[stationary_index] - 1.0) > 1e-10:
        raise ValueError("No eigenvalue sufficiently close to 1 was found.")
    stationary_vector = eigenvectors[:, stationary_index]
    if np.max(np.abs(stationary_vector.imag)) > 1e-10:
        raise ValueError("The stationary eigenvector has a significant imaginary part.")
    pagerank_vector = stationary_vector.real.copy()
    if pagerank_vector.sum() < 0:
        pagerank_vector *= -1
    if np.min(pagerank_vector) < -1e-10:
        raise ValueError("The stationary eigenvector contains mixed signs.")
    pagerank_vector = np.maximum(pagerank_vector, 0.0)
    pagerank_vector /= np.linalg.norm(pagerank_vector, ord=1)
    return eigenvalues, pagerank_vector


def print_eigenvalues(eigenvalues):
    """Retain complex parts: a real transition matrix can have complex eigenvalues."""
    print("Eigenvalues of the transposed matrix:")
    for index, value in enumerate(eigenvalues, start=1):
        print(f"  {index:2d}: {value.real:.10f}{value.imag:+.10f}j")


def power_pagerank(matrix, tolerance=1e-10, max_iterations=100000):
    """Iterate a uniform probability row vector until its L1 change is small.

    NumPy's one-dimensional vector has no explicit row/column orientation;
    vector @ matrix implements the row-vector update requested here. The L1
    norm is the sum of absolute differences, not the Euclidean norm. A finite
    iteration limit reports nonconvergence instead of looping indefinitely.
    """
    page_count = matrix.shape[0]
    previous_vector = np.ones(page_count) / page_count
    for iteration in range(1, max_iterations + 1):
        next_vector = previous_vector @ matrix
        change = np.linalg.norm(next_vector - previous_vector, ord=1)
        if change <= tolerance:
            return next_vector, iteration, change
        previous_vector = next_vector
    raise RuntimeError("Power method did not converge within the iteration limit.")


def report_power_method(matrix, eigen_vector, page_labels):
    """Show convergence and agreement with the independently computed eigenvector."""
    power_vector, iterations, change = power_pagerank(matrix)
    print(f"Iterations: {iterations}")
    print(f"Final successive-iterate L1 difference: {change:.12e}")
    print("Converged PageRank vector:")
    print_vector(power_vector, page_labels)
    print(f"L1 difference from eigendecomposition: "
          f"{np.linalg.norm(power_vector - eigen_vector, ord=1):.12e}")


def main():
    """Execute all nine assignment sections in order using the verified input."""
    # ============================================================
    # Section 1: Build Raw Google Matrix G
    # ============================================================
    print_banner(1, "Raw Google Matrix G")
    page_labels = list("ABCDEFGHIJKL")
    page_count = len(page_labels)
    page_indices = {page: index for index, page in enumerate(page_labels)}
    # The prompt's adjacency list is authoritative; each entry is an outlink.
    adjacency_list = {
        "A": ["B", "D"], "B": ["D", "E"], "C": ["B", "A", "H"],
        "D": ["E", "F"], "E": ["D", "G", "C", "H"],
        "F": ["A", "D", "G", "K"], "G": ["D", "I"],
        "H": ["G", "J"], "I": ["F", "G", "K", "J"],
        "J": ["I", "L"], "K": ["I", "L"], "L": [],
    }
    adjacency_matrix = np.zeros((page_count, page_count))
    for source_page, destination_pages in adjacency_list.items():
        for destination_page in destination_pages:
            adjacency_matrix[page_indices[source_page], page_indices[destination_page]] = 1.0
    # Divide only nonempty rows; this leaves dangling rows at zero and avoids
    # division by zero. [:, None] gives each row its own broadcast divisor.
    outlink_counts = adjacency_matrix.sum(axis=1)
    google_matrix = np.zeros_like(adjacency_matrix)
    linked_rows = outlink_counts > 0
    google_matrix[linked_rows] = (
        adjacency_matrix[linked_rows] / outlink_counts[linked_rows, None]
    )
    print_matrix(google_matrix, page_labels, page_labels)
    print_row_sums(google_matrix, page_labels)

    # ============================================================
    # Section 2: Fix Dangling Nodes
    # ============================================================
    print_banner(2, "Fix Dangling Nodes")
    # A surfer leaving a dangling page chooses uniformly among all pages,
    # including the current page. All subsequent uses of G refer to this fix.
    dangling_rows = google_matrix.sum(axis=1) == 0
    print("Dangling pages: " + ", ".join(np.array(page_labels)[dangling_rows]))
    google_matrix[dangling_rows] = 1.0 / page_count
    print_matrix(google_matrix, page_labels, page_labels)
    print_row_sums(google_matrix, page_labels)

    # ============================================================
    # Section 3: Eigenvalue Decomposition of G^T
    # ============================================================
    print_banner(3, "Eigendecomposition of G^T")
    eigenvalues, pagerank_vector = eigen_pagerank(google_matrix)
    print_eigenvalues(eigenvalues)
    print("PageRank vector:")
    print_vector(pagerank_vector, page_labels)
    print_ranking(pagerank_vector, page_labels)

    # ============================================================
    # Section 4: Power Method
    # ============================================================
    print_banner(4, "Power Method for G")
    report_power_method(google_matrix, pagerank_vector, page_labels)

    # ============================================================
    # Section 5: Teleportation Matrix G_tilde
    # ============================================================
    print_banner(5, "Teleportation Matrix G_tilde")
    # With probability alpha follow G; otherwise choose any page uniformly.
    # Positive teleportation makes the chain irreducible and aperiodic,
    # guaranteeing a unique stationary distribution and power convergence.
    damping_factor = 0.85
    uniform_matrix = np.ones((page_count, page_count)) / page_count
    teleportation_matrix = damping_factor * google_matrix + (1 - damping_factor) * uniform_matrix
    print(f"alpha = {damping_factor:.2f}")
    print_matrix(teleportation_matrix, page_labels, page_labels)
    print_row_sums(teleportation_matrix, page_labels)

    # ============================================================
    # Section 6: Eigendecomposition of G_tilde^T
    # ============================================================
    print_banner(6, "Eigendecomposition of G_tilde^T")
    teleportation_eigenvalues, teleportation_pagerank = eigen_pagerank(teleportation_matrix)
    print_eigenvalues(teleportation_eigenvalues)
    print("PageRank vector with teleportation:")
    print_vector(teleportation_pagerank, page_labels)
    print_ranking(teleportation_pagerank, page_labels)
    # Compare positions rather than page-aligned scores to expose rank changes.
    print(f"\n{'Position':>8} {'G page':>8} {'G score':>14} {'G_tilde page':>14} {'G_tilde score':>15}")
    for position, (old_index, new_index) in enumerate(zip(
        ranking_indices(pagerank_vector), ranking_indices(teleportation_pagerank)
    ), start=1):
        print(f"{position:8d} {page_labels[old_index]:>8} {pagerank_vector[old_index]:14.10f} "
              f"{page_labels[new_index]:>14} {teleportation_pagerank[new_index]:15.10f}")

    # ============================================================
    # Section 7: Power Method for G_tilde
    # ============================================================
    print_banner(7, "Power Method for G_tilde")
    report_power_method(teleportation_matrix, teleportation_pagerank, page_labels)

    # ============================================================
    # Section 8: Term-Document Matrix and Queries
    # ============================================================
    print_banner(8, "Term-Document Matrix T")
    # Commas separate brands so that 'Alfa Romeo' remains one complete term.
    keywords_by_page = {
        "A": "Dodge, Nissan, Mercedes, Audi, Jaguar, Hyundai, Porsche, Lexus, Mazda",
        "B": "Dodge, Mercedes, Jaguar, Hyundai, Toyota, Tesla, Cadillac, Subaru",
        "C": "Nissan, Mercedes, Jaguar, Toyota, Porsche, Mazda, Volkswagen",
        "D": "Mercedes, Hyundai, Tesla, Porsche, Lexus, Alfa Romeo, Subaru",
        "E": "Dodge, Jaguar, Toyota, Porsche, Cadillac, Volkswagen, Subaru",
        "F": "Nissan, Mercedes, Tesla, Porsche, Mazda, Alfa Romeo, Volkswagen",
        "G": "Audi, Jaguar, Tesla, Porsche, Lexus, Mazda, Alfa Romeo",
        "H": "Audi, Hyundai, Toyota, Cadillac, Lexus, Alfa Romeo, Volkswagen",
        "I": "Dodge, Nissan, Jaguar, Hyundai, Toyota, Cadillac, Volkswagen",
        "J": "Audi, Toyota, Porsche, Cadillac, Lexus, Mazda, Alfa Romeo, Subaru",
        "K": "Jaguar, Hyundai, Tesla, Porsche, Cadillac, Mazda, Alfa Romeo, Subaru",
        "L": "Nissan, Mercedes, Audi, Jaguar, Toyota, Tesla, Lexus, Mazda, Subaru",
    }
    keyword_sets = {page: set(brands.split(", ")) for page, brands in keywords_by_page.items()}
    terms = sorted(set().union(*keyword_sets.values()))
    term_indices = {term: index for index, term in enumerate(terms)}
    # Orientation is 15 terms by 12 documents, despite the source data being
    # organized by webpage. Binary entries record presence, not frequency.
    term_doc_matrix = np.array([
        [int(term in keyword_sets[page]) for page in page_labels] for term in terms
    ], dtype=int)
    print(f"Dimensions: {len(terms)} terms x {page_count} documents")
    print_matrix(term_doc_matrix, terms, page_labels, integer_entries=True)

    # ============================================================
    # Section 9: Search Queries
    # ============================================================
    print_banner(9, "Search Queries")
    queries = [
        ("SINGLE", "Nissan", ["Nissan"]),
        ("AND", "Mercedes AND Jaguar", ["Mercedes", "Jaguar"]),
        ("OR", "Toyota OR Porsche", ["Toyota", "Porsche"]),
        ("BUT NOT", "Mazda BUT NOT Volkswagen", ["Mazda", "Volkswagen"]),
    ]
    for operation, query_text, query_terms in queries:
        print(f"\nQuery ({operation}): {query_text}")
        keyword_matches = []
        for term in query_terms:
            # A separate one-hot q for each keyword selects its row of T via
            # d^T = q^T @ T. Keep these vectors separate for Boolean filtering.
            query_vector = np.zeros(len(terms), dtype=int)
            query_vector[term_indices[term]] = 1
            document_scores = query_vector @ term_doc_matrix
            keyword_matches.append(document_scores != 0)
            print(f"  q for {term} (terms in the Section 8 order): "
                  + " ".join(str(int(value)) for value in query_vector))
            print(f"  d for {term}: " + ", ".join(
                f"{page}={int(score)}" for page, score in zip(page_labels, document_scores)
            ))
        # Elementwise &, |, and ~ implement intersection, union, and exclusion.
        # Summing keyword scores would lose which particular keyword matched.
        if operation == "SINGLE":
            match_mask = keyword_matches[0]
        elif operation == "AND":
            match_mask = keyword_matches[0] & keyword_matches[1]
        elif operation == "OR":
            match_mask = keyword_matches[0] | keyword_matches[1]
        else:
            match_mask = keyword_matches[0] & ~keyword_matches[1]
        matching_indices = np.flatnonzero(match_mask)
        ranked_matches = matching_indices[
            np.argsort(-teleportation_pagerank[matching_indices], kind="stable")
        ]
        print("Matching pages: " + ", ".join(page_labels[index] for index in matching_indices))
        print("Ranking by G_tilde PageRank: " + " > ".join(page_labels[index] for index in ranked_matches))
        for position, index in enumerate(ranked_matches, start=1):
            print(f"  {position}. {page_labels[index]}: {teleportation_pagerank[index]:.10f}")


# Run the assignment when invoked as a script; importing keeps helpers usable
# for independent numerical checks without printing the entire submission.
if __name__ == "__main__":
    main()
