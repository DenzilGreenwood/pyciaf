"""
FOCUSED CIAF Framework Analysis Tool
Analyzes ONLY the core CIAF codebase for consolidation opportunities
"""

import os
import ast
import sys
import re
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path


class FocusedCIAFAnalyzer:
    """Analyze only the core CIAF framework code for consolidation"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.analysis = {
            'classes': defaultdict(list),
            'methods': defaultdict(list), 
            'functions': defaultdict(list),
            'variables': defaultdict(list),
            'imports': defaultdict(list),
            'file_summary': {},
            'ciaf_components': {
                'model_wrappers': [],
                'regressors': [],
                'utilities': [],
                'lcm_components': [],
                'base_classes': [],
                'frameworks': []
            }
        }
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze only the core CIAF codebase"""
        print("🔍 FOCUSED CIAF FRAMEWORK ANALYSIS")
        print("=" * 50)
        
        # Focus on core CIAF directories only
        ciaf_dirs = [
            "ciaf/",
            "models/",
            "tools/",
            "tests/"
        ]
        
        # Exclude directories to avoid noise
        exclude_patterns = [
            "venv/", ".venv/", "__pycache__/", ".git/",
            "site-packages/", "node_modules/", "dist/",
            ".pytest_cache/", "build/", "egg-info/"
        ]
        
        python_files = []
        
        for ciaf_dir in ciaf_dirs:
            dir_path = self.root_path / ciaf_dir
            if dir_path.exists():
                for file_path in dir_path.rglob("*.py"):
                    # Skip files matching exclude patterns
                    rel_path = file_path.relative_to(self.root_path)
                    if not any(pattern in str(rel_path) for pattern in exclude_patterns):
                        python_files.append(file_path)
        
        print(f"📁 Found {len(python_files)} Python files in core CIAF codebase")
        
        # Analyze each file
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                print(f"⚠️ Error analyzing {file_path}: {e}")
        
        # Perform CIAF-specific analysis
        self._analyze_ciaf_components()
        self._generate_consolidation_recommendations()
        
        return self.analysis
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Relative path for reporting
            rel_path = file_path.relative_to(self.root_path)
            
            file_analysis = {
                'classes': [],
                'methods': [],
                'functions': [],
                'variables': [],
                'imports': [],
                'lines_of_code': len(content.splitlines()),
                'file_path': str(rel_path),
                'docstring': ast.get_docstring(tree)
            }
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                self._analyze_node(node, file_analysis, str(rel_path))
            
            self.analysis['file_summary'][str(rel_path)] = file_analysis
            
        except Exception as e:
            print(f"❌ Failed to analyze {file_path}: {e}")
    
    def _analyze_node(self, node: ast.AST, file_analysis: Dict, file_path: str):
        """Analyze individual AST nodes"""
        
        if isinstance(node, ast.ClassDef):
            class_info = {
                'name': node.name,
                'file': file_path,
                'line': node.lineno,
                'methods': [],
                'base_classes': [self._get_name(base) for base in node.bases],
                'docstring': ast.get_docstring(node),
                'is_ciaf_class': 'ciaf' in node.name.lower() or 'lcm' in node.name.lower()
            }
            
            # Extract methods from the class
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_info = {
                        'name': item.name,
                        'args': [arg.arg for arg in item.args.args],
                        'is_private': item.name.startswith('_'),
                        'docstring': ast.get_docstring(item)
                    }
                    class_info['methods'].append(method_info)
            
            self.analysis['classes'][node.name].append(class_info)
            file_analysis['classes'].append(class_info)
            
        elif isinstance(node, ast.FunctionDef):
            # Only capture standalone functions (not methods)
            # Simple heuristic: if function is at module level, it's not a method
            is_method = hasattr(node, 'parent_class')  # This would need custom visitor
            
            # For now, we'll capture all functions and filter later if needed
            if True:  # We'll include all functions for now
                func_info = {
                    'name': node.name,
                    'file': file_path,
                    'line': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'is_private': node.name.startswith('_'),
                    'docstring': ast.get_docstring(node),
                    'is_ciaf_function': 'ciaf' in node.name.lower() or 'lcm' in node.name.lower()
                }
                
                self.analysis['functions'][node.name].append(func_info)
                file_analysis['functions'].append(func_info)
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    import_info = {
                        'module': f"{node.module}.{alias.name}" if alias.name != '*' else node.module,
                        'alias': alias.asname,
                        'file': file_path,
                        'line': node.lineno
                    }
                    
                    self.analysis['imports'][node.module].append(import_info)
                    file_analysis['imports'].append(import_info)
    
    def _get_name(self, node):
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _analyze_ciaf_components(self):
        """Categorize CIAF-specific components"""
        
        for class_name, occurrences in self.analysis['classes'].items():
            name_lower = class_name.lower()
            
            # Categorize classes
            if 'wrapper' in name_lower:
                self.analysis['ciaf_components']['model_wrappers'].append({
                    'name': class_name,
                    'files': [occ['file'] for occ in occurrences],
                    'methods_count': sum(len(occ['methods']) for occ in occurrences)
                })
                
            elif 'regressor' in name_lower or 'regression' in name_lower:
                self.analysis['ciaf_components']['regressors'].append({
                    'name': class_name,
                    'files': [occ['file'] for occ in occurrences],
                    'methods_count': sum(len(occ['methods']) for occ in occurrences)
                })
                
            elif 'utils' in name_lower or 'utility' in name_lower or 'helper' in name_lower:
                self.analysis['ciaf_components']['utilities'].append({
                    'name': class_name,
                    'files': [occ['file'] for occ in occurrences],
                    'methods_count': sum(len(occ['methods']) for occ in occurrences)
                })
                
            elif 'lcm' in name_lower:
                self.analysis['ciaf_components']['lcm_components'].append({
                    'name': class_name,
                    'files': [occ['file'] for occ in occurrences],
                    'methods_count': sum(len(occ['methods']) for occ in occurrences)
                })
                
            elif any(base in name_lower for base in ['base', 'abstract', 'interface']):
                self.analysis['ciaf_components']['base_classes'].append({
                    'name': class_name,
                    'files': [occ['file'] for occ in occurrences],
                    'methods_count': sum(len(occ['methods']) for occ in occurrences)
                })
                
            elif 'framework' in name_lower or 'ciaf' in name_lower:
                self.analysis['ciaf_components']['frameworks'].append({
                    'name': class_name,
                    'files': [occ['file'] for occ in occurrences],
                    'methods_count': sum(len(occ['methods']) for occ in occurrences)
                })
    
    def _generate_consolidation_recommendations(self):
        """Generate specific consolidation recommendations for CIAF"""
        
        recommendations = {
            'duplicate_classes': [],
            'naming_inconsistencies': [],
            'consolidation_opportunities': [],
            'architectural_improvements': []
        }
        
        # Find similar classes
        class_names = list(self.analysis['classes'].keys())
        
        # Group similar class names
        similar_groups = defaultdict(list)
        for name in class_names:
            # Normalize name for comparison
            normalized = re.sub(r'(CIAF|Ciaf|ciaf)', '', name).lower()
            normalized = re.sub(r'(Model|model)', '', normalized)
            normalized = re.sub(r'[^a-z]', '', normalized)
            
            if len(normalized) > 3:  # Only meaningful names
                similar_groups[normalized].append(name)
        
        # Find groups with multiple classes
        for normalized, group in similar_groups.items():
            if len(group) > 1:
                recommendations['duplicate_classes'].append({
                    'concept': normalized,
                    'classes': group,
                    'suggestion': f'Consider consolidating {group} into a single class'
                })
        
        # Analyze naming inconsistencies
        ciaf_classes = [name for name in class_names if 'ciaf' in name.lower()]
        
        # Check for inconsistent prefixes
        prefixes = set()
        for name in ciaf_classes:
            if name.startswith('CIAF'):
                prefixes.add('CIAF')
            elif name.startswith('Ciaf'):
                prefixes.add('Ciaf')
            elif name.startswith('ciaf'):
                prefixes.add('ciaf')
        
        if len(prefixes) > 1:
            recommendations['naming_inconsistencies'].append({
                'issue': 'Inconsistent CIAF class prefixes',
                'prefixes': list(prefixes),
                'suggestion': 'Standardize on single prefix (recommend "CIAF")'
            })
        
        # Consolidation opportunities
        components = self.analysis['ciaf_components']
        
        if len(components['model_wrappers']) > 1:
            recommendations['consolidation_opportunities'].append({
                'component': 'Model Wrappers',
                'count': len(components['model_wrappers']),
                'items': components['model_wrappers'],
                'suggestion': 'Consider consolidating into single CIAFModelWrapper base class'
            })
        
        if len(components['regressors']) > 3:
            recommendations['consolidation_opportunities'].append({
                'component': 'Regressors',
                'count': len(components['regressors']),
                'items': components['regressors'],
                'suggestion': 'Create CIAFRegressionBase class with common functionality'
            })
        
        # Architectural improvements
        if len(components['utilities']) > 5:
            recommendations['architectural_improvements'].append({
                'issue': f"Too many utility classes ({len(components['utilities'])})",
                'suggestion': 'Consolidate utilities into modules: data_utils, wrapper_utils, error_utils'
            })
        
        if not components['base_classes']:
            recommendations['architectural_improvements'].append({
                'issue': 'No base classes found',
                'suggestion': 'Create abstract base classes for common CIAF patterns'
            })
        
        self.analysis['recommendations'] = recommendations
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate focused CIAF analysis report"""
        
        report = []
        report.append("🔍 FOCUSED CIAF FRAMEWORK ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append("📊 CIAF CODEBASE OVERVIEW")
        report.append("-" * 30)
        report.append(f"Files analyzed: {len(self.analysis['file_summary'])}")
        report.append(f"Classes found: {len(self.analysis['classes'])}")
        report.append(f"Functions found: {len(self.analysis['functions'])}")
        report.append("")
        
        # File breakdown
        report.append("📁 FILE ANALYSIS")
        report.append("-" * 30)
        sorted_files = sorted(self.analysis['file_summary'].items(), 
                            key=lambda x: x[1]['lines_of_code'], reverse=True)
        
        for i, (file_path, info) in enumerate(sorted_files[:10]):
            classes = len(info['classes'])
            functions = len(info['functions'])
            report.append(f"{i+1:2d}. {file_path}")
            report.append(f"    Lines: {info['lines_of_code']}, Classes: {classes}, Functions: {functions}")
        
        report.append("")
        
        # CIAF Components Analysis
        report.append("🎯 CIAF COMPONENTS BREAKDOWN")
        report.append("-" * 30)
        
        components = self.analysis['ciaf_components']
        
        for component_type, items in components.items():
            if items:
                report.append(f"{component_type.replace('_', ' ').title()}:")
                for item in items:
                    report.append(f"  • {item['name']} ({item['methods_count']} methods)")
                    report.append(f"    Files: {', '.join(item['files'])}")
                report.append("")
        
        # Class Analysis
        report.append("🏗️ CLASS ANALYSIS")
        report.append("-" * 30)
        
        # Most complex classes
        class_complexity = {}
        for class_name, occurrences in self.analysis['classes'].items():
            total_methods = sum(len(occ['methods']) for occ in occurrences)
            class_complexity[class_name] = total_methods
        
        sorted_classes = sorted(class_complexity.items(), key=lambda x: x[1], reverse=True)
        
        report.append("Most Complex Classes (by method count):")
        for i, (class_name, method_count) in enumerate(sorted_classes[:10]):
            files = [occ['file'] for occ in self.analysis['classes'][class_name]]
            report.append(f"{i+1:2d}. {class_name} ({method_count} methods)")
            report.append(f"    Files: {', '.join(set(files))}")
        
        report.append("")
        
        # Method Analysis
        report.append("⚙️ METHOD ANALYSIS")
        report.append("-" * 30)
        
        # Extract all methods from all classes
        all_methods = []
        for occurrences in self.analysis['classes'].values():
            for occ in occurrences:
                all_methods.extend(occ['methods'])
        
        # Count method name frequencies
        method_counts = defaultdict(int)
        for method in all_methods:
            method_counts[method['name']] += 1
        
        sorted_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
        
        report.append("Most Common Method Names:")
        for i, (method_name, count) in enumerate(sorted_methods[:10]):
            if count > 1:  # Only show duplicated methods
                report.append(f"{i+1:2d}. {method_name} ({count} occurrences)")
        
        report.append("")
        
        # Function Analysis
        report.append("🔧 FUNCTION ANALYSIS")
        report.append("-" * 30)
        
        function_counts = {name: len(occurrences) for name, occurrences in self.analysis['functions'].items()}
        sorted_functions = sorted(function_counts.items(), key=lambda x: x[1], reverse=True)
        
        report.append("Most Common Function Names:")
        for i, (func_name, count) in enumerate(sorted_functions[:10]):
            if count > 1:
                report.append(f"{i+1:2d}. {func_name} ({count} occurrences)")
        
        report.append("")
        
        # Recommendations
        if 'recommendations' in self.analysis:
            recommendations = self.analysis['recommendations']
            
            report.append("💡 CONSOLIDATION RECOMMENDATIONS")
            report.append("-" * 40)
            
            if recommendations['duplicate_classes']:
                report.append("🔄 Potential Duplicate Classes:")
                for dup in recommendations['duplicate_classes']:
                    report.append(f"  • Concept: {dup['concept']}")
                    report.append(f"    Classes: {', '.join(dup['classes'])}")
                    report.append(f"    Suggestion: {dup['suggestion']}")
                report.append("")
            
            if recommendations['naming_inconsistencies']:
                report.append("📝 Naming Inconsistencies:")
                for inconsistency in recommendations['naming_inconsistencies']:
                    report.append(f"  • Issue: {inconsistency['issue']}")
                    report.append(f"    Prefixes found: {', '.join(inconsistency['prefixes'])}")
                    report.append(f"    Suggestion: {inconsistency['suggestion']}")
                report.append("")
            
            if recommendations['consolidation_opportunities']:
                report.append("🎯 Consolidation Opportunities:")
                for opp in recommendations['consolidation_opportunities']:
                    report.append(f"  • Component: {opp['component']} ({opp['count']} items)")
                    report.append(f"    Suggestion: {opp['suggestion']}")
                report.append("")
            
            if recommendations['architectural_improvements']:
                report.append("🏗️ Architectural Improvements:")
                for improvement in recommendations['architectural_improvements']:
                    report.append(f"  • Issue: {improvement['issue']}")
                    report.append(f"    Suggestion: {improvement['suggestion']}")
                report.append("")
        
        # Implementation Plan
        report.append("📋 IMPLEMENTATION PLAN")
        report.append("-" * 30)
        report.append("Phase 1: Immediate Consolidation")
        report.append("  1. Create ciaf/utils/ directory with:")
        report.append("     - data_utils.py (data handling functions)")
        report.append("     - wrapper_utils.py (wrapper helper functions)")
        report.append("     - error_utils.py (error handling utilities)")
        report.append("  2. Create base classes:")
        report.append("     - ciaf/core/base_model.py (abstract base for all models)")
        report.append("     - ciaf/core/base_regressor.py (base for regression models)")
        report.append("")
        report.append("Phase 2: Class Consolidation")
        report.append("  1. Merge similar wrapper classes into CIAFModelWrapper")
        report.append("  2. Standardize naming conventions (use 'CIAF' prefix)")
        report.append("  3. Remove duplicate functionality")
        report.append("")
        report.append("Phase 3: Architecture Refinement")
        report.append("  1. Create consistent interface patterns")
        report.append("  2. Implement plugin architecture for extensions")
        report.append("  3. Add comprehensive documentation")
        report.append("")
        
        report.append("=" * 60)
        report.append("🎯 ANALYSIS COMPLETE")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_file}")
        
        return report_text


def main():
    """Main analysis function"""
    
    # Set up paths
    current_dir = Path(__file__).parent
    root_path = current_dir.parent  # Go up to PYPI root
    
    print(f"🔍 Analyzing CORE CIAF codebase at: {root_path}")
    print()
    
    # Create analyzer
    analyzer = FocusedCIAFAnalyzer(str(root_path))
    
    # Run analysis
    results = analyzer.analyze_codebase()
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save detailed outputs
    output_dir = current_dir / "analysis_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Save text report
    report_file = output_dir / "focused_ciaf_analysis_report.txt"
    analyzer.generate_report(str(report_file))
    
    # Save JSON data
    json_file = output_dir / "focused_ciaf_analysis_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print()
    print("📁 Focused analysis outputs saved:")
    print(f"  📄 Text Report: {report_file}")
    print(f"  📊 JSON Data: {json_file}")
    
    return results


if __name__ == "__main__":
    main()