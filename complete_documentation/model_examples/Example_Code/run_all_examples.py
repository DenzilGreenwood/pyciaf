#!/usr/bin/env python3
"""
CIAF Example Runner
Executes all AI model implementation examples and provides a comprehensive summary.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any

# Set UTF-8 encoding for Windows compatibility
import codecs
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def run_example(example_path: str, example_name: str) -> Dict[str, Any]:
    """Run a single example and capture results."""
    print(f"\n{'='*60}")
    print(f"Running {example_name} Example")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Get the directory containing the example file
        example_dir = os.path.dirname(example_path)
        example_filename = os.path.basename(example_path)
        
        # Run the example from its own directory with proper environment
        result = subprocess.run(
            [sys.executable, example_filename], 
            cwd=example_dir,  # Set working directory to the example's directory
            capture_output=True, 
            text=True,
            encoding='utf-8',  # Force UTF-8 encoding
            errors='replace',  # Replace invalid characters
            timeout=300,  # 5 minute timeout
            env=os.environ.copy()  # Inherit current environment
        )
        
        execution_time = time.time() - start_time
        
        # Check if execution was successful - prioritize exit code
        basic_success = result.returncode == 0
        
        # Try to detect success indicators if we have readable output
        output_success = False
        completion_found = False
        if result.stdout:
            try:
                stdout_lines = result.stdout.split('\n')
                # Look for ASCII completion sentinel
                completion_found = any("IMPLEMENTATION_COMPLETE" in line for line in stdout_lines)
                key_features_found = any("Key" in line and ("Features Demonstrated" in line or "Generation Features" in line) for line in stdout_lines)
                many_checkmarks = len([line for line in stdout_lines if "" in line]) >= 10
                
                output_success = (completion_found or key_features_found or 
                               any("" in line and "Complete!" in line for line in stdout_lines) or
                               many_checkmarks)
            except Exception as e:
                # If there are encoding issues in parsing, just use basic success
                output_success = False
        
        # Success is determined by exit code only - be strict
        success = (result.returncode == 0)
        
        # Extract key metrics from output (handle encoding errors gracefully)
        output_lines = []
        if result.stdout:
            try:
                output_lines = result.stdout.split('\n')
            except:
                output_lines = []
        
        error_lines = []
        if result.stderr:
            try:
                error_lines = result.stderr.split('\n')
            except:
                error_lines = []
        
        # Look for completion message - use ASCII sentinel
        completed = completion_found
        
        # Count key features demonstrated using ASCII bullets
        features_count = 0
        in_features_section = False
        for line in output_lines:
            s = line.strip()
            if "Features Demonstrated" in s:
                in_features_section = True
                continue
            if in_features_section:
                if s.startswith(("- ", "* ", "[OK]", "[X]", "+")):
                    features_count += 1
                elif s == "" or "Troubleshooting" in s or "---" in s:
                    in_features_section = False
        
        # Look for compliance information
        compliance_mentions = len([line for line in output_lines if any(keyword in line.lower() for keyword in ['compliance', 'audit', 'bias', 'fairness', 'privacy', 'hipaa', 'gdpr'])])
        
        return {
            'name': example_name,
            'success': success,
            'completed': completed,
            'execution_time': execution_time,
            'features_demonstrated': features_count,
            'compliance_mentions': compliance_mentions,
            'output_lines': len(output_lines),
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return {
            'name': example_name,
            'success': False,
            'completed': False,
            'execution_time': execution_time,
            'features_demonstrated': 0,
            'compliance_mentions': 0,
            'output_lines': 0,
            'stdout': '',
            'stderr': 'Execution timed out after 5 minutes',
            'returncode': -1
        }
    
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'name': example_name,
            'success': False,
            'completed': False,
            'execution_time': execution_time,
            'features_demonstrated': 0,
            'compliance_mentions': 0,
            'output_lines': 0,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def print_summary(results: List[Dict[str, Any]]):
    """Print a comprehensive summary of all example runs."""
    print(f"\n{'='*80}")
    print("CIAF EXAMPLES EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Examples: {len(results)}")
    
    # Overall statistics
    successful_runs = sum(1 for r in results if r['success'])
    completed_runs = sum(1 for r in results if r['completed'])
    total_execution_time = sum(r['execution_time'] for r in results)
    total_features = sum(r['features_demonstrated'] for r in results)
    total_compliance = sum(r['compliance_mentions'] for r in results)
    
    print(f"\nOverall Statistics:")
    print(f"   Successful Executions: {successful_runs}/{len(results)} ({successful_runs/len(results)*100:.1f}%)")
    print(f"   Completed Examples: {completed_runs}/{len(results)} ({completed_runs/len(results)*100:.1f}%)")
    print(f"   Total Execution Time: {total_execution_time:.1f} seconds")
    print(f"   Features Demonstrated: {total_features}")
    print(f"   Compliance Mentions: {total_compliance}")
    
    # Individual example results
    print(f"\nIndividual Example Results:")
    print(f"{'Example':<25} {'Status':<10} {'Time':<8} {'Features':<10} {'Compliance':<12}")
    print("-" * 75)
    
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        features = f"{result['features_demonstrated']} items"
        compliance = f"{result['compliance_mentions']} mentions"
        time_str = f"{result['execution_time']:.1f}s"
        
        print(f"{result['name']:<25} {status:<10} {time_str:<8} {features:<10} {compliance:<12}")
    
    # Detailed analysis
    print(f"\nDetailed Analysis:")
    
    for result in results:
        print(f"\n{result['name']} Example:")
        print(f"   Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"   Execution Time: {result['execution_time']:.1f} seconds")
        print(f"   Output Lines: {result['output_lines']}")
        print(f"   Return Code: {result['returncode']}")
        
        if result['success']:
            print(f"   Features: {result['features_demonstrated']} demonstrated")
            print(f"   Compliance: {result['compliance_mentions']} mentions")
        else:
            # Show full error message for debugging
            print(f"   Return Code: {result['returncode']}")
            if result['stderr']:
                error_preview = result['stderr'][:500]  # Show more error details
                print(f"   Error (stderr): {error_preview}")
                if len(result['stderr']) > 500:
                    print(f"   ... (error truncated, full length: {len(result['stderr'])} chars)")
            if result['stdout']:
                # Show some stdout even for failed runs to help debug
                stdout_preview = result['stdout'][:300]
                print(f"   Output (stdout): {stdout_preview}")
                if len(result['stdout']) > 300:
                    print(f"   ... (output truncated, full length: {len(result['stdout'])} chars)")
            if not result['stderr'] and not result['stdout']:
                print(f"   Error: No output captured - possible encoding or subprocess issue")
    
    # Recommendations
    print(f"\nRecommendations:")
    
    if successful_runs == len(results):
        print("   All examples executed successfully!")
        print("   Consider installing full dependencies for enhanced functionality")
        print("   Review individual outputs for detailed insights")
    else:
        failed_examples = [r['name'] for r in results if not r['success']]
        print(f"   {len(failed_examples)} examples failed: {', '.join(failed_examples)}")
        print("   Check dependencies and error messages")
        print("   Refer to README.md for troubleshooting")
    
    print(f"\nFor detailed documentation, see: README.md")
    print(f"For CIAF package information, visit: https://github.com/your-repo/ciaf")

def main():
    """Main execution function."""
    print("CIAF AI Model Implementation Examples Runner")
    print("=" * 60)
    print("This script will execute all AI model examples and provide a comprehensive summary.")
    print("Each example demonstrates CIAF integration with different AI model types.")
    
    # Define examples to run
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples = [
        {
            'name': 'LLM',
            'path': os.path.join(current_dir, 'LLM_example', 'llm_example.py'),
            'description': 'Large Language Model with bias monitoring'
        },
        {
            'name': 'Classifier',
            'path': os.path.join(current_dir, 'Classifier_example', 'classifier_example.py'),
            'description': 'Classification model with fairness assessment'
        },
        {
            'name': 'CNN',
            'path': os.path.join(current_dir, 'CNN_example', 'cnn_example.py'),
            'description': 'CNN for medical imaging with privacy protection'
        },
        {
            'name': 'Diffusion',
            'path': os.path.join(current_dir, 'Diffusion_example', 'diffusion_example.py'),
            'description': 'Diffusion model for content generation'
        },
        {
            'name': 'Agentic System',
            'path': os.path.join(current_dir, 'Agentic_System_example', 'agentic_system_example.py'),
            'description': 'Multi-agent system with governance'
        }
    ]
    
    print(f"\nExamples to execute:")
    for i, example in enumerate(examples, 1):
        print(f"   {i}. {example['name']}: {example['description']}")
        if not os.path.exists(example['path']):
            print(f"      Warning: File not found: {example['path']}")
    
    # Check if user wants to continue
    try:
        response = input(f"\nExecute all {len(examples)} examples? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Execution cancelled.")
            return
    except KeyboardInterrupt:
        print("\nExecution cancelled.")
        return
    
    # Execute all examples
    results = []
    start_time = time.time()
    
    for example in examples:
        if os.path.exists(example['path']):
            result = run_example(example['path'], example['name'])
            results.append(result)
        else:
            print(f"\nSkipping {example['name']}: File not found")
            results.append({
                'name': example['name'],
                'success': False,
                'completed': False,
                'execution_time': 0,
                'features_demonstrated': 0,
                'compliance_mentions': 0,
                'output_lines': 0,
                'stdout': '',
                'stderr': 'File not found',
                'returncode': -1
            })
    
    total_time = time.time() - start_time
    
    # Print comprehensive summary
    print_summary(results)
    
    print(f"\nTotal Execution Time: {total_time:.1f} seconds")
    print("All examples completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        print("\nThank you for exploring CIAF examples!")