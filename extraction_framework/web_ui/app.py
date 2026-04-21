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

from extraction_framework.test_runner import TestRunner
from extraction_framework.model_loader import ModelLoader
from extraction_framework.ground_truth import GroundTruthManager
from extraction_framework.extractors import get_all_extractors, get_extractor_by_name
from extraction_framework.llm_providers import get_available_providers
from extraction_framework.page_validator import PageValidator, load_validation_rules_from_model
from extraction_framework.scoring import ExtractionResult

# Load environment
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize components
BASE_DIR = Path(__file__).parent.parent.parent
TEST_DIR = BASE_DIR / "Test"
RESULTS_DIR = BASE_DIR / "extraction_framework" / "results"
GT_DIR = BASE_DIR / "extraction_framework" / "ground_truth"

model_loader = ModelLoader(TEST_DIR)
runner = TestRunner(results_dir=RESULTS_DIR, ground_truth_dir=GT_DIR, model_loader=model_loader)
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
            "test_folders": model_loader.list_test_folders(),
            "providers_config": get_all_providers_config()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/document-files')
@app.route('/api/pdf-files')
def get_document_files():
    """Get list of available supported PDF files, optionally filtered by test folder"""
    try:
        test_folder_filter = request.args.get('test_folder')
        document_files = []
        
        for test_folder in TEST_DIR.glob("*"):
            if not test_folder.is_dir():
                continue
            
            # Skip if filter is specified and doesn't match
            if test_folder_filter and test_folder.name != test_folder_filter:
                continue
                
            for document_file in test_folder.iterdir():
                if not document_file.is_file() or document_file.suffix.lower() != ".pdf":
                    continue

                document_files.append({
                    "name": document_file.name,
                    "path": str(document_file),
                    "test_folder": test_folder.name,
                    "size": document_file.stat().st_size,
                    "extension": ".pdf"
                })

        return jsonify({"files": document_files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-test', methods=['POST'])
def run_test():
    """Run extraction test with detailed logging"""
    import traceback
    
    try:
        data = request.json
        print(f"\n{'='*60}")
        print(f"[API] Received test request")
        print(f"[API] PDF: {data.get('pdf_file')}")
        print(f"[API] Extractor: {data.get('extractor')}")
        print(f"[API] Provider: {data.get('llm_provider')}")
        print(f"[API] Model: {data.get('llm_model')}")
        
        pdf_file = Path(data['pdf_file'])
        extractor = data['extractor']
        provider = data['llm_provider']
        model = data.get('llm_model')
        model_file = data.get('model_file', 'modello.py')
        
        # Load extraction model and scoring rules
        test_folder = pdf_file.parent.name
        print(f"[API] Loading model from test folder: {test_folder}")
        extraction_model = model_loader.get_model_for_test(test_folder, model_file)
        extraction_model_module = model_loader.get_module_for_test(test_folder, model_file)
        print(f"[API] Model loaded: {extraction_model.__name__ if extraction_model else 'None'}")
        
        # Load scoring rules for this test folder
        print(f"[API] Loading scoring rules for: {test_folder}")
        runner.set_scoring_rules_for_test(test_folder)
        
        # Run extraction (API key from environment variables)
        print(f"[API] Starting extraction...")
        result = runner.run_extraction(
            pdf_path=pdf_file,
            extractor_name=extractor,
            llm_provider=provider,
            llm_model=model,
            extraction_model=extraction_model,
            extraction_model_module=extraction_model_module
        )
        
        print(f"[API] Extraction completed. Success: {result.success}")
        if result.extracted_data:
            print(f"[API] Data extracted: {len(str(result.extracted_data))} chars")
        if result.error:
            print(f"[API] Error: {result.error}")
        
        # Save result to disk for comparison feature
        try:
            saved_path = runner.scorer.save_result(result)
            print(f"[API] Result saved to: {saved_path}")
        except Exception as save_error:
            print(f"[API] Warning: Failed to save result: {save_error}")
        
        print(f"{'='*60}\n")
        
        # Convert to dict and ensure JSON serializable
        response_data = result.model_dump()
        return jsonify(response_data)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[API] ERROR: {e}")
        print(f"[API] Traceback:\n{error_trace}")
        return jsonify({
            "error": str(e),
            "traceback": error_trace,
            "success": False,
            "extractor_name": data.get('extractor', 'unknown'),
            "llm_provider": data.get('llm_provider', 'unknown'),
            "llm_model": data.get('llm_model', 'unknown'),
            "extraction_time": 0
        }), 200  # Return 200 to avoid parse errors


@app.route('/api/pdf/<test_folder>/<pdf_name>')
def serve_pdf(test_folder, pdf_name):
    """Serve PDF file"""
    try:
        pdf_path = TEST_DIR / test_folder / pdf_name
        print(f"[API] Serving PDF: {pdf_path}")
        if not pdf_path.exists():
            print(f"[API] ERROR: PDF not found: {pdf_path}")
            return jsonify({"error": "PDF not found"}), 404
        return send_file(pdf_path, mimetype='application/pdf')
    except Exception as e:
        print(f"[API] ERROR serving PDF: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/comparison/<path:pdf_path>')
def get_comparison(pdf_path):
    """Get comparison of extraction results for a PDF"""
    try:
        # Extract just the filename from the path
        pdf_name = pdf_path.split('/')[-1].split('\\')[-1]
        # Use generic stem for any supported document extension
        pdf_stem = Path(pdf_name).stem
        print(f"[API] Comparison request for: {pdf_path}")
        print(f"[API] PDF name: {pdf_name}, stem: {pdf_stem}")
        
        # Load all results for this PDF from results directory
        # Results are saved as: <pdf_stem>_<extractor>_<provider>_<timestamp>.json
        results = []
        available_files = list(RESULTS_DIR.glob("*.json"))
        print(f"[API] Searching in: {RESULTS_DIR}")
        print(f"[API] Available result files: {[f.name for f in available_files]}")
        
        for result_file in available_files:
            try:
                # Check if this result file is for our PDF (match by stem)
                if result_file.name.startswith(pdf_stem):
                    print(f"[API] Found matching file by stem: {result_file.name}")
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        # Also verify the pdf_file field matches
                        result_pdf_name = result_data.get('pdf_file', '').split('\\')[-1].split('/')[-1]
                        print(f"[API]   Result PDF name: {result_pdf_name}")
                        if result_pdf_name == pdf_name or Path(result_pdf_name).stem == pdf_stem:
                            results.append(result_data)
                            print(f"[API]   ✓ Loaded result from: {result_file.name}")
                        else:
                            print(f"[API]   ✗ PDF name mismatch: {result_pdf_name} != {pdf_name}")
            except Exception as e:
                print(f"[API] Error loading result file {result_file}: {e}")
                import traceback
                traceback.print_exc()
        
        if not results:
            print(f"[API] ❌ No results found for {pdf_name} (stem: {pdf_stem})")
            return jsonify({
                "pdf_file": pdf_name,
                "total_extractions": 0,
                "successful_extractions": 0,
                "avg_extraction_time": 0,
                "total_tokens": 0,
                "avg_total_tokens": 0,
                "avg_input_tokens": 0,
                "avg_output_tokens": 0,
                "results": []
            })
        
        print(f"[API] Found {len(results)} results for {pdf_name}")
        
        # Calculate statistics
        successful = [r for r in results if r.get('success', False)]
        total_time = sum(r.get('extraction_time', 0) for r in results)
        avg_time = total_time / len(results) if results else 0
        
        # Calculate token statistics
        total_input_tokens = sum(r.get('input_tokens', 0) for r in successful if r.get('input_tokens'))
        total_output_tokens = sum(r.get('output_tokens', 0) for r in successful if r.get('output_tokens'))
        total_tokens = sum(r.get('total_tokens', 0) for r in successful if r.get('total_tokens'))
        avg_input_tokens = total_input_tokens / len(successful) if successful else 0
        avg_output_tokens = total_output_tokens / len(successful) if successful else 0
        avg_total_tokens = total_tokens / len(successful) if successful else 0
        
        comparison = {
            "pdf_file": pdf_name,
            "total_extractions": len(results),
            "successful_extractions": len(successful),
            "avg_extraction_time": avg_time,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_total_tokens": avg_total_tokens,
            "results": results
        }
        
        if successful:
            # Find best extraction (fastest successful one)
            comparison["best_extraction"] = min(successful, key=lambda x: x.get('extraction_time', float('inf')))
        
        return jsonify(comparison)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[API] ERROR in comparison: {e}")
        print(f"[API] Traceback:\n{error_trace}")
        return jsonify({"error": str(e), "traceback": error_trace}), 500


@app.route('/api/results/clear-failed', methods=['POST'])
def clear_failed_results():
    """Delete all failed extraction results from history"""
    try:
        print(f"[API] Clearing failed results from {RESULTS_DIR}")
        deleted_count = 0
        deleted_files = []
        
        for result_file in RESULTS_DIR.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    
                if not result_data.get('success', False):
                    deleted_files.append(result_file.name)
                    result_file.unlink()
                    deleted_count += 1
            except Exception as e:
                print(f"[API] Error processing {result_file}: {e}")
        
        print(f"[API] Deleted {deleted_count} failed results")
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[API] ERROR in clear_failed_results: {e}")
        return jsonify({"error": str(e), "traceback": error_trace}), 500


@app.route('/api/comparisons/all')
def get_all_comparisons():
    """Get comparison summary for all PDFs with results"""
    try:
        print(f"[API] Loading all comparisons from {RESULTS_DIR}")
        
        # Group results by PDF
        pdf_results = {}
        for result_file in RESULTS_DIR.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    # Add filename for timestamp extraction
                    result_data['_filename'] = result_file.name
                    pdf_file = result_data.get('pdf_file', '')
                    pdf_name = pdf_file.split('\\')[-1].split('/')[-1]
                    
                    if pdf_name not in pdf_results:
                        pdf_results[pdf_name] = []
                    pdf_results[pdf_name].append(result_data)
            except Exception as e:
                print(f"[API] Error loading {result_file}: {e}")
        
        print(f"[API] Found results for {len(pdf_results)} PDFs")
        
        # Global analytics across ALL tests
        global_extractor_stats = {}
        global_llm_stats = {}
        global_stack_stats = {}
        
        # Build comparison summary for each PDF
        comparisons = []
        for pdf_name, results in pdf_results.items():
            successful = [r for r in results if r.get('success', False)]
            
            if not successful:
                continue
            
            # Use only the latest results per technology stack
            stack_latest = {}
            for result in successful:
                stack_key = f"{result.get('extractor_name', '')}|{result.get('llm_provider', '')}|{result.get('llm_model', '')}"
                filename = result.get('_filename', '')
                if stack_key not in stack_latest or filename > stack_latest[stack_key].get('_filename', ''):
                    stack_latest[stack_key] = result
            
            successful_latest = list(stack_latest.values())
            
            # Load Ground Truth for this PDF
            pdf_full_path = successful_latest[0].get('pdf_file', '')
            ground_truth = gt_manager.load_ground_truth(pdf_full_path)
            
            # Generate comparison using Ground Truth if available
            extraction_results = [ExtractionResult(**r) for r in successful_latest]
            comparison_obj = runner.scorer.compare_results(extraction_results, ground_truth=ground_truth)
            
            # Calculate scores for UI display
            scored_results = []
            for result in successful_latest:
                score = runner.scorer._calculate_result_score(
                    result.get('extracted_data', {}),
                    comparison_obj.field_scores,
                    is_ground_truth=bool(ground_truth)
                )
                
                scored_results.append({
                    'extractor': result.get('extractor_name', 'Unknown'),
                    'provider': result.get('llm_provider', 'Unknown'),
                    'model': result.get('llm_model', 'Unknown'),
                    'score': score,
                    'records': len(result.get('extracted_data', {}).get('consumi', result.get('extracted_data', {}).get('fatture', result.get('extracted_data', {}).get('rifiuti', [])))),
                    'time': result.get('extraction_time', 0),
                    'tokens': result.get('total_tokens', 0),
                    'success': True,
                    'extracted_data': result.get('extracted_data'),
                })
            
            scored_results.sort(key=lambda x: (-x['score'], x['time']))
            
            # Assign ranks
            current_rank = 1
            for i, stack in enumerate(scored_results):
                if i > 0 and stack['score'] == scored_results[i-1]['score']:
                    stack['rank'] = scored_results[i-1]['rank']
                else:
                    stack['rank'] = current_rank
                current_rank += 1
            
            comparisons.append({
                'pdf_name': pdf_name,
                'total_extractions': len(results),
                'successful_extractions': len(successful),
                'total_records': comparison_obj.total_extractions, # Placeholder for total records info
                'best_stack': ' / '.join([f"{s['extractor']}+{s['provider']}" for s in scored_results if s['rank'] == 1]),
                'best_score': scored_results[0]['score'] if scored_results else 0,
                'winners_count': len([s for s in scored_results if s['rank'] == 1]),
                'all_stacks': scored_results,
                'has_ground_truth': bool(ground_truth)
            })
            
            # Accumulate global statistics for each stack tested
            for stack in scored_results:
                # Extractor stats
                ext = stack['extractor']
                if ext not in global_extractor_stats:
                    global_extractor_stats[ext] = {'scores': [], 'times': [], 'tokens': [], 'ranks': [], 'wins': 0}
                global_extractor_stats[ext]['scores'].append(stack['score'])
                global_extractor_stats[ext]['times'].append(stack['time'])
                global_extractor_stats[ext]['tokens'].append(stack['tokens'])
                global_extractor_stats[ext]['ranks'].append(stack['rank'])
                if stack['rank'] == 1:
                    global_extractor_stats[ext]['wins'] += 1
                
                # LLM stats
                llm_key = f"{stack['provider']}/{stack['model']}"
                if llm_key not in global_llm_stats:
                    global_llm_stats[llm_key] = {'scores': [], 'times': [], 'tokens': [], 'ranks': [], 'wins': 0}
                global_llm_stats[llm_key]['scores'].append(stack['score'])
                global_llm_stats[llm_key]['times'].append(stack['time'])
                global_llm_stats[llm_key]['tokens'].append(stack['tokens'])
                global_llm_stats[llm_key]['ranks'].append(stack['rank'])
                if stack['rank'] == 1:
                    global_llm_stats[llm_key]['wins'] += 1
                
                # Complete stack stats
                stack_key = f"{stack['extractor']}+{stack['provider']}/{stack['model']}"
                if stack_key not in global_stack_stats:
                    global_stack_stats[stack_key] = {'scores': [], 'times': [], 'tokens': [], 'ranks': [], 'wins': 0}
                global_stack_stats[stack_key]['scores'].append(stack['score'])
                global_stack_stats[stack_key]['times'].append(stack['time'])
                global_stack_stats[stack_key]['tokens'].append(stack['tokens'])
                global_stack_stats[stack_key]['ranks'].append(stack['rank'])
                if stack['rank'] == 1:
                    global_stack_stats[stack_key]['wins'] += 1
        
        # Compute global analytics with normalized metrics
        global_extractor_analytics = []
        for ext, stats in global_extractor_stats.items():
            tests = len(stats['scores'])
            wins = stats['wins']
            
            # Win rate: percentage of tests won
            win_rate = (wins / tests * 100) if tests > 0 else 0
            
            # Average rank (lower is better)
            avg_rank = sum(stats['ranks']) / tests if tests > 0 else 999
            
            # Normalized score: combines win rate with inverse avg rank
            # Formula: (win_rate * 0.6) + ((1 / avg_rank) * 40)
            # This gives 60% weight to win rate, 40% to ranking performance
            normalized_score = (win_rate * 0.6) + ((1 / avg_rank) * 40) if avg_rank > 0 else 0
            
            global_extractor_analytics.append({
                'name': ext,
                'avg_score': sum(stats['scores']) / tests if tests > 0 else 0,
                'max_score': max(stats['scores']) if stats['scores'] else 0,
                'min_score': min(stats['scores']) if stats['scores'] else 0,
                'avg_time': sum(stats['times']) / tests if tests > 0 else 0,
                'avg_tokens': sum(stats['tokens']) / tests if tests > 0 else 0,
                'avg_rank': avg_rank,
                'wins': wins,
                'tests': tests,
                'win_rate': win_rate,
                'normalized_score': normalized_score
            })
        global_extractor_analytics.sort(key=lambda x: -x['normalized_score'])
        
        global_llm_analytics = []
        for llm, stats in global_llm_stats.items():
            tests = len(stats['scores'])
            wins = stats['wins']
            win_rate = (wins / tests * 100) if tests > 0 else 0
            avg_rank = sum(stats['ranks']) / tests if tests > 0 else 999
            normalized_score = (win_rate * 0.6) + ((1 / avg_rank) * 40) if avg_rank > 0 else 0
            global_llm_analytics.append({
                'name': llm,
                'avg_score': sum(stats['scores']) / tests if tests > 0 else 0,
                'max_score': max(stats['scores']) if stats['scores'] else 0,
                'min_score': min(stats['scores']) if stats['scores'] else 0,
                'avg_time': sum(stats['times']) / tests if tests > 0 else 0,
                'avg_tokens': sum(stats['tokens']) / tests if tests > 0 else 0,
                'avg_rank': avg_rank,
                'wins': wins,
                'tests': tests,
                'win_rate': win_rate,
                'normalized_score': normalized_score
            })
        global_llm_analytics.sort(key=lambda x: -x['normalized_score'])
        
        global_stack_analytics = []
        for stack_key, stats in global_stack_stats.items():
            tests = len(stats['scores'])
            wins = stats['wins']
            win_rate = (wins / tests * 100) if tests > 0 else 0
            avg_rank = sum(stats['ranks']) / tests if tests > 0 else 999
            normalized_score = (win_rate * 0.6) + ((1 / avg_rank) * 40) if avg_rank > 0 else 0
            global_stack_analytics.append({
                'name': stack_key,
                'avg_score': sum(stats['scores']) / tests if tests > 0 else 0,
                'max_score': max(stats['scores']) if stats['scores'] else 0,
                'min_score': min(stats['scores']) if stats['scores'] else 0,
                'avg_time': sum(stats['times']) / tests if tests > 0 else 0,
                'avg_tokens': sum(stats['tokens']) / tests if tests > 0 else 0,
                'avg_rank': avg_rank,
                'wins': wins,
                'tests': tests,
                'win_rate': win_rate,
                'normalized_score': normalized_score
            })
        global_stack_analytics.sort(key=lambda x: -x['normalized_score'])
        
        # Sort by PDF name
        comparisons.sort(key=lambda x: x['pdf_name'])
        
        return jsonify({
            'comparisons': comparisons,
            'global_extractor_analytics': global_extractor_analytics,
            'global_llm_analytics': global_llm_analytics,
            'global_stack_analytics': global_stack_analytics
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[API] ERROR in get_all_comparisons: {e}")
        print(f"[API] Traceback:\n{error_trace}")
        return jsonify({"error": str(e), "traceback": error_trace}), 500


@app.route('/api/stack-data/<path:pdf_name>/<extractor>/<provider>')
def get_stack_data(pdf_name, extractor, provider):
    """Get extracted data for a specific stack (extractor + provider combination)"""
    try:
        print(f"[API] Getting stack data for: {pdf_name} - {extractor} + {provider}")
        
        pdf_stem = Path(pdf_name).stem
        
        # Search for matching result file
        # Format: {pdf_stem}_{extractor}_{provider}_{timestamp}.json
        matching_files = []
        for result_file in RESULTS_DIR.glob(f"{pdf_stem}_{extractor}_{provider}_*.json"):
            matching_files.append(result_file)
        
        if not matching_files:
            print(f"[API] No result file found for {pdf_stem}_{extractor}_{provider}")
            print(f"[API] Available files: {list(RESULTS_DIR.glob('*.json'))}")
            return jsonify({"error": "Result not found"}), 404
        
        # Get the most recent result (last in sorted list)
        matching_files.sort()
        latest_file = matching_files[-1]
        
        print(f"[API] Loading data from: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        return jsonify(result_data)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[API] ERROR in get_stack_data: {e}")
        print(f"[API] Traceback:\n{error_trace}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/ground-truth/<test_folder>/<pdf_name>', methods=['GET'])
def get_ground_truth(test_folder, pdf_name):
    """Get ground truth for a PDF"""
    try:
        pdf_path = TEST_DIR / test_folder / pdf_name.replace('.json', '.pdf')
        ground_truth = gt_manager.load_ground_truth(str(pdf_path))
        
        if not ground_truth:
            return jsonify({"error": "Ground truth not found"}), 404
            
        return jsonify(ground_truth)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ground-truth/<test_folder>/<pdf_name>', methods=['POST'])
def save_ground_truth_api(test_folder, pdf_name):
    """Save ground truth for a PDF"""
    try:
        data = request.json
        pdf_path = TEST_DIR / test_folder / pdf_name.replace('.json', '.pdf')
        
        gt_manager.save_ground_truth(
            pdf_file=str(pdf_path),
            data=data,
            validated_by=request.args.get('user', 'web_ui'),
            save_to_test_folder=True
        )
        
        return jsonify({"success": True, "message": "Ground truth saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/test-regex', methods=['POST'])
def test_regex():
    """Test regex patterns on a PDF"""
    import traceback
    try:
        data = request.json
        print(f"\n{'='*60}")
        print(f"[API] Testing regex patterns")
        
        pdf_file = Path(data['pdf_file'])
        test_folder = pdf_file.parent.name
        extractor_name = data.get('extractor', 'PyMuPDF-XML')
        
        print(f"[API] PDF: {pdf_file}")
        print(f"[API] Test folder: {test_folder}")
        print(f"[API] Extractor: {extractor_name}")
        
        # Load model module to get validation rules
        extraction_model_module = model_loader.get_module_for_test(test_folder)
        
        from extraction_framework.page_validator import load_validation_rules_from_model, PageValidator
        
        # Get validation rules
        validation_rules = load_validation_rules_from_model(extraction_model_module)
        
        if not validation_rules:
            return jsonify({
                "error": "No PAGE_VALIDATION_RULES found in model",
                "rules": [],
                "pages": []
            }), 400
        
        # Extract pages from PDF
        extractor = get_extractor_by_name(extractor_name)
        pages = extractor.extract_pages(pdf_file)
        
        print(f"[API] Extracted {len(pages)} pages")
        print(f"[API] Validation rules: {len(validation_rules)}")
        
        # Create validator and test each page
        validator = PageValidator(validation_rules)
        
        page_results = []
        for i, page_text in enumerate(pages):
            is_valid = validator.validate_page(page_text)
            
            # Find which rules matched
            matched_rules = []
            for rule_idx, rule in enumerate(validator.validation_rules):
                all_match = all(
                    pattern.search(page_text) is not None
                    for pattern in rule['patterns']
                )
                if all_match:
                    matched_rules.append({
                        "index": rule_idx,
                        "description": rule['description']
                    })
            
            page_results.append({
                "page_number": i + 1,
                "is_valid": is_valid,
                "matched_rules": matched_rules,
                "text_preview": page_text[:500] + "..." if len(page_text) > 500 else page_text,
                "text_length": len(page_text)
            })
        
        # Calculate filtering results
        filtered_pages, filter_stats = validator.filter_pages(pages, verbose=False)
        
        response = {
            "success": True,
            "rules": [
                {
                    "index": i,
                    "description": r.get('description', 'Unnamed rule'),
                    "patterns": [str(p.pattern) for p in r['patterns']]
                }
                for i, r in enumerate(validator.validation_rules)
            ],
            "pages": page_results,
            "filter_stats": filter_stats,
            "total_pages": len(pages),
            "valid_pages": sum(1 for p in page_results if p['is_valid']),
            "filtered_pages": len(filtered_pages)
        }
        
        print(f"[API] Valid pages: {response['valid_pages']}/{response['total_pages']}")
        print(f"[API] Pages after filtering: {response['filtered_pages']}")
        print(f"{'='*60}\n")
        
        return jsonify(response)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[API] ERROR: {e}")
        print(f"[API] Traceback:\n{error_trace}")
        return jsonify({"error": str(e), "traceback": error_trace}), 500


@app.route('/api/test-regex-batch', methods=['POST'])
def test_regex_batch():
    """Test regex validation on ALL PDFs in ALL test folders"""
    try:
        import fitz  # PyMuPDF
        
        data = request.get_json()
        
        print(f"\n{'='*60}")
        print(f"[API] Batch Regex Test")
        print(f"{'='*60}")
        
        # Get all test folders
        test_folders = [d for d in TEST_DIR.iterdir() if d.is_dir()]
        
        all_results = []
        total_pages_before = 0
        total_pages_after = 0
        total_pages_removed = 0
        
        for test_folder in test_folders:
            folder_name = test_folder.name
            print(f"\n[Folder] {folder_name}")
            
            # Check if modello.py exists
            model_file = test_folder / "modello.py"
            if not model_file.exists():
                print(f"  ⚠️  No modello.py found, skipping")
                continue
            
            # Load validation rules
            try:
                # Load the model module dynamically
                import importlib.util
                spec = importlib.util.spec_from_file_location(f"modello_{folder_name}", model_file)
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                
                validation_rules = load_validation_rules_from_model(model_module)
                if not validation_rules:
                    print(f"  ⚠️  No PAGE_VALIDATION_RULES found")
                    continue
                
                validator = PageValidator(validation_rules)
                print(f"  ✓ Loaded {len(validation_rules)} validation rules")
            except Exception as e:
                print(f"  ❌ Error loading rules: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Get all PDFs in this folder (case-insensitive on Windows)
            pdf_files = list(set(test_folder.glob("*.pdf")) | set(test_folder.glob("*.PDF")))
            print(f"  Found {len(pdf_files)} PDFs")
            
            for pdf_path in pdf_files:
                pdf_name = pdf_path.name
                print(f"    Testing: {pdf_name}")
                
                try:
                    # Open PDF with PyMuPDF directly
                    doc = fitz.open(str(pdf_path))
                    total_pages = len(doc)
                    
                    # Validate each page
                    pages_to_keep = []
                    valid_pages = 0
                    
                    for page_num in range(total_pages):
                        page = doc[page_num]
                        page_text = page.get_text()
                        
                        if validator.validate_page(page_text):
                            pages_to_keep.append(page_num)
                            valid_pages += 1
                    
                    doc.close()
                    
                    # Calculate filtering stats (remove invalid from head/tail, keep middle)
                    if pages_to_keep:
                        first_valid = min(pages_to_keep)
                        last_valid = max(pages_to_keep)
                        filtered_pages = list(range(first_valid, last_valid + 1))
                    else:
                        filtered_pages = list(range(total_pages))
                        first_valid = 0
                        last_valid = total_pages - 1
                    
                    removed_head = first_valid
                    removed_tail = total_pages - last_valid - 1
                    
                    filter_stats = {
                        'total_pages': total_pages,
                        'validated_pages': valid_pages,
                        'filtered_pages': len(filtered_pages),
                        'removed_from_head': removed_head,
                        'removed_from_tail': removed_tail,
                        'text_reduction_percent': ((total_pages - len(filtered_pages)) / total_pages * 100) if total_pages > 0 else 0
                    }
                    
                    result = {
                        'test_folder': folder_name,
                        'pdf_file': pdf_name,
                        'total_pages': total_pages,
                        'valid_pages': valid_pages,
                        'filter_stats': filter_stats
                    }
                    
                    all_results.append(result)
                    
                    total_pages_before += total_pages
                    total_pages_after += len(filtered_pages)
                    total_pages_removed += removed_head + removed_tail
                    
                    print(f"      ✓ {len(filtered_pages)}/{total_pages} pages kept (-{filter_stats['text_reduction_percent']:.1f}%)")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    continue
        
        # Calculate summary statistics
        total_pdfs = len(all_results)
        avg_reduction = sum(r['filter_stats']['text_reduction_percent'] for r in all_results) / total_pdfs if total_pdfs > 0 else 0
        avg_validation_rate = sum(r['valid_pages'] / r['total_pages'] * 100 for r in all_results) / total_pdfs if total_pdfs > 0 else 0
        
        summary = {
            'total_pdfs': total_pdfs,
            'total_pages_before': total_pages_before,
            'total_pages_after': total_pages_after,
            'total_pages_removed': total_pages_removed,
            'avg_reduction_percent': avg_reduction,
            'avg_validation_rate': avg_validation_rate
        }
        
        print(f"\n{'='*60}")
        print(f"[Summary]")
        print(f"  Total PDFs: {total_pdfs}")
        print(f"  Total pages: {total_pages_before} → {total_pages_after} (-{total_pages_removed})")
        print(f"  Avg reduction: {avg_reduction:.1f}%")
        print(f"  Avg validation: {avg_validation_rate:.1f}%")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'summary': summary,
            'results': all_results
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[API] ERROR: {e}")
        print(f"[API] Traceback:\n{error_trace}")
        return jsonify({"error": str(e), "traceback": error_trace}), 500


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
