"""Example: Exploring the source domain library.

This example shows how to explore the available domains
and their structures for inspiration.
"""

from x_creative.core.domain_loader import DomainLibrary


def main(target_domain: str = "open_source_development") -> None:
    """Explore the domain library for a target domain."""

    library = DomainLibrary.from_target_domain(target_domain)

    print("=" * 60)
    print(f"X-Creative: Source Domain Library ({target_domain})")
    print("=" * 60)
    print(f"\nTotal domains: {len(library)}")
    print()

    # List all domains
    print("Available Domains:")
    print("-" * 40)
    for domain in library:
        print(f"  - {domain.id}: {domain.name} ({len(domain.structures)} structures)")

    # Deep dive into a specific domain
    print("\n" + "=" * 60)
    print("Deep Dive: Queueing Theory")
    print("=" * 60)

    queueing = library.get("queueing_theory")
    if queueing:
        print(f"\n{queueing.description}\n")

        print("Structures:")
        for s in queueing.structures:
            print(f"\n  {s.id} ({s.name})")
            print(f"    {s.description}")
            print(f"    Key variables: {', '.join(s.key_variables)}")
            print(f"    Dynamics: {s.dynamics}")

        print("\nTarget Mappings:")
        for m in queueing.target_mappings:
            print(f"\n  {m.structure} -> {m.target}")
            print(f"    Observable: {m.observable}")

    # Statistics
    print("\n" + "=" * 60)
    print("Library Statistics")
    print("=" * 60)

    total_structures = sum(len(d.structures) for d in library)
    total_mappings = sum(len(d.target_mappings) for d in library)

    print(f"\nTotal structures: {total_structures}")
    print(f"Total target mappings: {total_mappings}")
    print(f"Average structures per domain: {total_structures / len(library):.1f}")


if __name__ == "__main__":
    main()
