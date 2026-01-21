"""
app/cli.py

Command-line interface for RAG operations.
Simplified using LangChain components.
"""

import argparse
import sys
from pathlib import Path

from llama_rag.service import RAGService
from llama_rag.config import get_config
from llama_rag.utils import get_logger, configure_logging

logger = get_logger(__name__)


def ingest_command(args):
    """
    Ingest documents into vector store.
    
    Usage:
        python -m app.cli ingest examples/demo-corpus/ --output ./data/index
    """
    print(f"\n📥 Ingesting documents from: {', '.join(args.sources)}")
    logger.info(f"Ingesting from: {args.sources}")
    
    # Read config once
    config = get_config()
    
    # Create service with config passed down
    output_path = Path(args.output)
    service = RAGService(
        index_path=output_path,
        auto_load=False,
        auto_save=False,
        # Config values flow from here
    )
    
    # Process each source
    for source in args.sources:
        source_path = Path(source)
        if not source_path.exists():
            logger.error(f"Source not found: {source}")
            continue
        
        logger.info(f"Processing: {source}")
        service.ingest_from_directory(
            source_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )
    
    # Save explicitly
    service.save()
    
    stats = service.get_stats()
    print(f"✅ Ingestion complete! Index saved to: {output_path}")
    if args.verbose:
        print(f"   Stats: {stats}")
    logger.info(f"✓ Ingestion complete. Index saved to: {output_path}")


def query_command(args):
    """
    Query the RAG system.
    
    Usage:
        python -m app.cli query "What is machine learning?" --index ./data/index
    """
    index_path = Path(args.index)
    
    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        logger.error("Run 'ingest' command first to create an index")
        sys.exit(1)
    
    # Read config once
    config = get_config()
    
    # Load service with config passed down
    service = RAGService(
        index_path=index_path,
        auto_load=True,
        auto_save=False,
        # Config values flow from here
    )
    
    # Query
    logger.info(f"Query: {args.query}")
    print(f"\n🔍 Querying: {args.query}")
    logger.debug(f"Retrieving top {args.k or service.retrieval_k} documents")
    
    response = service.query(args.query, k=args.k)
    
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(response)
    print("="*80 + "\n")
    
    # Show retrieved documents if requested
    if args.show_docs:
        docs = service.retrieve(args.query, k=args.k)
        print("\nRETRIEVED DOCUMENTS:")
        print("="*80)
        for i, doc in enumerate(docs, 1):
            print(f"\n[{i}] {doc.metadata.get('source', 'Unknown')}")
            print(f"{doc.page_content[:200]}...")
        print("="*80 + "\n")





def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="RAG CLI - Simplified with LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global verbose flag
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (show detailed process logs)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "sources",
        nargs="+",
        help="Source files or directories to ingest",
    )
    ingest_parser.add_argument(
        "--output",
        "-o",
        default="./data/index",
        help="Output directory for index (default: ./data/index)",
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size (defaults to config)",
    )
    ingest_parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Chunk overlap (defaults to config)",
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument(
        "query",
        help="Query string",
    )
    query_parser.add_argument(
        "--index",
        "-i",
        default="./data/index",
        help="Index directory (default: ./data/index)",
    )
    query_parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of documents to retrieve (defaults to config)",
    )
    query_parser.add_argument(
        "--show-docs",
        action="store_true",
        help="Show retrieved documents",
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Configure logging based on verbose flag
    configure_logging(verbose=args.verbose)
    
    # Execute command
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "query":
        query_command(args)


if __name__ == "__main__":
    main()
