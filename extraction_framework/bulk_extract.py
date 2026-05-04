import sys
from pathlib import Path

# Add the directory containing 'extraction_framework' to sys.path
# Since this script is inside 'extraction_framework', parent is the root dir.
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

try:
    from extraction_framework.extractors.markdown_extractor import MarkdownExtractor
    import extraction_framework.console
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def extract_all():
    test_dir = root_dir / "Test"
    debug_dir = root_dir / "debug_payloads"
    debug_dir.mkdir(exist_ok=True)
    
    extractor = MarkdownExtractor()
    
    # Find all PDFs recursively in the Test directory
    pdf_files = list(test_dir.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {test_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting extraction...")
    
    success_count = 0
    fail_count = 0

    for pdf_path in pdf_files:
        debug_file = debug_dir / f"{pdf_path.stem}_{extractor.name}.txt"
        
        print(f"Extracting: {pdf_path.name} ...")
        
        try:
            document_text = extractor.extract_text(pdf_path)
            
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(document_text)
            
            print(f"  [green]✓[/green] Saved to {debug_file.name}")
            success_count += 1
        except Exception as e:
            print(f"  [red]❌ Failed[/red] {pdf_path.name}: {e}")
            fail_count += 1

    print(f"\nExtraction complete!")
    print(f"Successfully extracted: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Outputs saved to: {debug_dir}")

if __name__ == "__main__":
    extract_all()
