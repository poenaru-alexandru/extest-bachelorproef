"""Flask web application for Document Extraction Tester"""
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Initialize global colorized print override
import extraction_framework.console

from extraction_framework.benchmark_runner import BenchmarkRunner
from extraction_framework.ground_truth import GroundTruthManager
from extraction_framework.extractors import get_all_extractors
from extraction_framework.llm_providers import get_available_providers
from extraction_framework.scoring import ExtractionResult

# Load environment
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize components
BASE_DIR = Path(__file__).parent.parent.parent
TEST_DIR = Path(__file__).parent.parent / "Test"
RESULTS_DIR = BASE_DIR / "extraction_framework" / "results"
GT_DIR = BASE_DIR / "extraction_framework" / "ground_truth"

runner = BenchmarkRunner(results_dir=RESULTS_DIR, ground_truth_dir=GT_DIR)
gt_manager = GroundTruthManager(GT_DIR)


@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')


@app.route('/api/config')
def get_config():
    """Get configuration options"""
    try:
        from extraction_framework.llm_providers import get_provider_models, get_all_providers_config
        
        extractors = [e.name for e in get_all_extractors()]
        providers = get_available_providers()
        
        # Get models for each provider from configuration
        models = {}
        for provider in providers:
            provider_models = get_provider_models(provider)
            models[provider] = provider_models if provider_models else ["default"]
        
        return jsonify({
            "extractors": extractors,
            "providers": providers,
            "models": models,
            "providers_config": get_all_providers_config()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/document-files')
@app.route('/api/pdf-files')
def get_document_files():
    """Get list of available supported PDF files from Test directory"""
    try:
        document_files = []
        
        for document_file in TEST_DIR.rglob("*.pdf"):
            if not document_file.is_file():
                continue

            document_files.append({
                "name": document_file.name,
                "path": str(document_file),
                "size": document_file.stat().st_size,
                "extension": ".pdf"
            })

        return jsonify({"files": document_files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-test', methods=['POST'])
def run_test():
    """Run extraction test(s) with optimized batching for local models"""
    import traceback
    
    try:
        data = request.json
        print(f"\n{'='*60}")
        print(f"[API] Received extraction request")
        
        # Normalize inputs to lists for batch processing
        pdf_paths = data.get('pdf_files', [data.get('pdf_file')])
        pdf_files = [Path(p) for p in pdf_paths if p]
        
        extractors = data.get('extractors', [data.get('extractor')])
        
        # Handle single or multiple model configs
        if 'llm_configs' in data:
            llm_configs = data['llm_configs']
        else:
            llm_configs = [{
                "provider": data.get('llm_provider'),
                "model": data.get('llm_model')
            }]
            
        print(f"[API] Batch size: {len(pdf_files)} files, {len(extractors)} extractors, {len(llm_configs)} models")
        
        # Run test suite (handles Model-Document batching internally for local)
        all_results_dict = runner.run_test_suite(
            pdf_files=pdf_files,
            extractors=extractors,
            llm_configs=llm_configs
        )
        
        # Flatten results for response
        flat_results = []
        for pdf_path_str, results in all_results_dict.items():
            for res in results:
                # Add score using the new rule-based logic
                res_dict = res.model_dump()
                if res.success and res.extracted_data:
                    res_dict['score'] = runner.scorer.validate_and_score(res.extracted_data)
                else:
                    res_dict['score'] = 0.0
                flat_results.append(res_dict)
        
        print(f"[API] Suite completed. Total results: {len(flat_results)}")
        print(f"{'='*60}\n")
        
        # If it was a single test request, return just the first result for compatibility
        if len(flat_results) == 1 and 'pdf_file' in data:
            return jsonify(flat_results[0])
            
        return jsonify({"results": flat_results, "success": True})
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[API] ERROR: {e}")
        print(f"[API] Traceback:\n{error_trace}")
        return jsonify({
            "error": str(e),
            "traceback": error_trace,
            "success": False
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Document Extraction Tester - Web UI")
    print("=" * 60)
    print(f"Test directory: {TEST_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Ground truth directory: {GT_DIR}")
    print()
    print("Available providers:", get_available_providers())
    print("Available extractors:", [e.name for e in get_all_extractors()])
    print()
    print("Starting server on http://localhost:5000")
    print("=" * 60)
    
    # Use debug=False for production, or set via environment variable
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
