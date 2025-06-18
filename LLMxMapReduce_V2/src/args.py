import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the EntirePipeline with specified input, output, and prompt files.")
    
    # Input/output configuration
    parser.add_argument("--topic", type=str, required=True,
        help="Research topic for automatic content retrieval (type: str, alternative to --input_file)")
    parser.add_argument("--description", type=str, required=False, default=None,
        help="Detailed description of research topic used for query generation (type: str, default: None)")
    parser.add_argument("--top_n", type=int, default=10, 
        help="Maximum number of references to retrieve (type: int, default: 10)")
    parser.add_argument("--input_file", type=str, 
        help="Absolute path to input file containing pre-collected data (type: str, alternative to --topic)")
    parser.add_argument("--output_file", type=str, required=True, 
        help="Path to save processed output results")
    parser.add_argument("--config_file", type=str, default='config/model_config.json', 
        help="Path to model configuration JSON file (default: config/model_config.json)")

    # Pipeline processing parameters
    parser.add_argument("--data_num", type=int, default=None, 
        help="Number of data items to process:\n"
             "- None/unspecified: Process all available data (default)\n"
             "- Positive N: Process first N items\n"
             "- Negative N: Process all except last N items\n"
             "Affects memory usage and processing time linearly")
    parser.add_argument("--parallel_num", type=int, default=1, 
        help="Number of parallel processing workers (default: 1)")

    # Content processing modes
    parser.add_argument("--digest_group_mode", type=str, choices=["random", "llm"], default="llm", 
        help="Method for grouping content: 'random' or 'llm' (default: 'llm')")
    parser.add_argument("--skeleton_group_size", type=int, default=3, 
        help="Number of content digests combined to create each skeleton section:\n"
             "- Higher values: Broader outlines (better for overviews)\n"
             "- Lower values: More detailed outlines (better for analysis)\n"
             "- Default 3 balances detail and performance\n"
             "Affects outline granularity and processing time (type: int, default: 3)")

    # Block processing control
    parser.add_argument("--block_count", type=int, default=0, 
        help="Number of complete processing cycles (0=unlimited):\n"
             "- Each cycle includes: grouping → skeleton → digests → refinement\n"
             "- Higher values: More refined results but longer processing\n"
             "- Lower values: Faster results but less refinement\n"
             "- Default 0 (unlimited) for maximum quality\n"
             "Use with --output_each_block to compare cycle results (type: int, default: 0)")
    parser.add_argument("--output_each_block", action="store_true", 
        help="Flag to save intermediate results after each refinement cycle:\n"
             "- When present: Saves progressive refinement states\n"
             "- When omitted: Only final results saved (default)\n"
             "- Useful for analyzing how results improve through processing")

    # Convolution parameters (for MCTS)
    parser.add_argument("--conv_layer", type=int, default=6, 
        help="Number of MCTS refinement layers (type: int, default: 6):\n"
             "- Each layer performs suggestion generation → evaluation → selection\n"
             "- Higher values (8-10): Deeper refinement, better quality, slower\n"
             "- Lower values (3-4): Faster processing, less refinement\n"
             "- Default 6 balances quality and performance\n"
             "Recommended: 6 for most cases, 8-10 for final runs, 3-4 for testing")
    parser.add_argument("--conv_kernel_width", type=int, default=3, 
        help="Context window size for suggestion generation (type: int, default: 3):\n"
             "- Higher values (4-5): More context-aware, consistent suggestions\n"
             "- Lower values (1-2): More creative/diverse suggestions\n"
             "- Default 3 balances context and creativity\n"
             "Recommended: 3 for most cases, 4-5 for technical topics, 2 for brainstorming")
    parser.add_argument("--conv_result_num", type=int, default=10, 
        help="Suggestions generated per refinement layer (type: int, default: 10):\n"
             "- Higher values (15-20): More options → better results but slower\n"
             "- Lower values (5-8): Faster processing but fewer options\n"
             "- Default 10 balances quality and performance\n"
             "Recommended: 10 normally, 15-20 for final runs, 5-8 for testing")
    parser.add_argument("--top_k", type=int, default=6, 
        help="Top suggestions retained per refinement layer (type: int, default: 6):\n"
             "- Higher values (8-10): Better quality/diversity but slower\n"
             "- Lower values (3-5): Faster processing with tighter focus\n"
             "- Default 6 balances quality and performance\n"
             "Recommended: 6 normally, 8-10 for final runs, 3-4 for testing")

    # Self-refinement parameters
    parser.add_argument("--self_refine_count", type=int, default=3, 
        help="Iterations of self-improvement (type: int, default: 3):\n"
             "- Higher values (4-5): More refined outputs but slower\n"
             "- Lower values (1-2): Faster processing with less polish\n"
             "- Default 3 provides good quality without excessive processing\n"
             "Recommended: 3 normally, 4-5 for final outputs, 1-2 for testing")
    parser.add_argument("--self_refine_best_of", type=int, default=3, 
        help="Candidate refinements evaluated per iteration (type: int, default: 3):\n"
             "- Higher values (4-5): Better quality through comparison but slower\n"
             "- Lower values (1-2): Faster processing with less optimization\n"
             "- Default 3 balances quality and performance\n"
             "Recommended: 3 normally, 4-5 for final outputs, 1-2 for testing")
    args = parser.parse_args()
    return args
