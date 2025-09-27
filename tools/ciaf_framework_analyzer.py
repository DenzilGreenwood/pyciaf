"""
CIAF Framework Analysis Tool
Comprehensive analysis of all classes, methods, functions, and variables
across the entire CIAF codebase to enable consolidation and cleanup.
"""

import os
import ast
import sys
import re
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path


class CIAFCodeAnalyzer:
    """Analyze the entire CIAF codebase for consolidation opportunities"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.analysis = {
            'classes': defaultdict(list),
            'methods': defaultdict(list), 
            'functions': defaultdict(list),
            'variables': defaultdict(list),
            'imports': defaultdict(list),
            'file_summary': {},
            'naming_patterns': {
                'verbose_names': [],
                'inconsistent_naming': [],
                'duplicate_functionality': []
            }
        }
        self.processed_files = []
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the entire CIAF codebase"""
        print("🔍 CIAF FRAMEWORK COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        # Find all Python files
        python_files = list(self.root_path.rglob("*.py"))
        print(f"📁 Found {len(python_files)} Python files")
        
        # Analyze each file
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                print(f"⚠️ Error analyzing {file_path}: {e}")
        
        # Perform consolidation analysis
        self._analyze_naming_patterns()
        self._identify_duplicate_functionality()
        self._generate_recommendations()
        
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
                'file_path': str(rel_path)
            }
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                self._analyze_node(node, file_analysis, str(rel_path))
            
            self.analysis['file_summary'][str(rel_path)] = file_analysis
            self.processed_files.append(str(rel_path))
            
        except Exception as e:
            print(f"❌ Failed to analyze {file_path}: {e}")
    
    def _analyze_node(self, node: ast.AST, file_analysis: Dict, file_path: str):
        """Analyze individual AST nodes"""
        
        if isinstance(node, ast.ClassDef):
            class_info = {
                'name': node.name,
                'file': file_path,
                'line': node.lineno,
                'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                'base_classes': [self._get_name(base) for base in node.bases],
                'docstring': ast.get_docstring(node)
            }
            
            self.analysis['classes'][node.name].append(class_info)
            file_analysis['classes'].append(class_info)
            
        elif isinstance(node, ast.FunctionDef):
            # Check if it's a method (inside a class) or standalone function
            is_method = any(isinstance(parent, ast.ClassDef) 
                          for parent in ast.walk(ast.parse(ast.dump(node))))
            
            func_info = {
                'name': node.name,
                'file': file_path,
                'line': node.lineno,
                'args': [arg.arg for arg in node.args.args],
                'is_method': is_method,
                'is_private': node.name.startswith('_'),
                'docstring': ast.get_docstring(node)
            }
            
            if is_method:
                self.analysis['methods'][node.name].append(func_info)
                file_analysis['methods'].append(func_info)
            else:
                self.analysis['functions'][node.name].append(func_info)
                file_analysis['functions'].append(func_info)
                
        elif isinstance(node, ast.Assign):
            # Analyze variable assignments
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_info = {
                        'name': target.id,
                        'file': file_path,
                        'line': node.lineno,
                        'is_constant': target.id.isupper(),
                        'is_private': target.id.startswith('_')
                    }
                    
                    self.analysis['variables'][target.id].append(var_info)
                    file_analysis['variables'].append(var_info)
                    
        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_info = {
                    'module': alias.name,
                    'alias': alias.asname,
                    'file': file_path,
                    'line': node.lineno,
                    'type': 'import'
                }
                
                self.analysis['imports'][alias.name].append(import_info)
                file_analysis['imports'].append(import_info)
                
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    import_info = {
                        'module': f"{node.module}.{alias.name}" if alias.name != '*' else node.module,
                        'alias': alias.asname,
                        'file': file_path,
                        'line': node.lineno,
                        'type': 'from_import'
                    }
                    
                    module_name = node.module or 'relative'
                    self.analysis['imports'][module_name].append(import_info)
                    file_analysis['imports'].append(import_info)
    
    def _get_name(self, node):
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _analyze_naming_patterns(self):
        """Analyze naming patterns for inconsistencies"""
        
        # Check for verbose names
        verbose_threshold = 30  # Characters
        
        for name_type in ['classes', 'methods', 'functions']:
            for name, occurrences in self.analysis[name_type].items():
                if len(name) > verbose_threshold:
                    self.analysis['naming_patterns']['verbose_names'].append({
                        'type': name_type,
                        'name': name,
                        'length': len(name),
                        'occurrences': len(occurrences),
                        'files': [occ['file'] for occ in occurrences]
                    })
        
        # Check for inconsistent naming patterns
        self._find_inconsistent_naming()
        
    def _find_inconsistent_naming(self):
        """Find inconsistent naming patterns"""
        
        # Look for similar function/method names with different patterns
        method_names = list(self.analysis['methods'].keys())
        function_names = list(self.analysis['functions'].keys())
        all_names = method_names + function_names
        
        # Group by semantic similarity
        naming_groups = defaultdict(list)
        
        for name in all_names:
            # Remove common prefixes/suffixes and group
            clean_name = re.sub(r'^(ciaf_|_|get_|set_|create_|generate_)', '', name.lower())
            clean_name = re.sub(r'(_with_.*|_and_.*|_for_.*)$', '', clean_name)
            
            if len(clean_name) > 3:  # Ignore very short names
                naming_groups[clean_name].append(name)
        
        # Find groups with multiple naming patterns
        for base_name, variants in naming_groups.items():
            if len(variants) > 1:
                # Check if they have significantly different naming styles
                patterns = set()
                for variant in variants:
                    if 'comprehensive' in variant or 'with_full' in variant:
                        patterns.add('verbose')
                    elif len(variant.split('_')) > 4:
                        patterns.add('descriptive')
                    else:
                        patterns.add('concise')
                
                if len(patterns) > 1:
                    self.analysis['naming_patterns']['inconsistent_naming'].append({
                        'base_concept': base_name,
                        'variants': variants,
                        'patterns': list(patterns)
                    })
    
    def _identify_duplicate_functionality(self):
        """Identify potentially duplicate functionality"""
        
        # Look for methods/functions with similar names
        similar_functions = defaultdict(list)
        
        all_funcs = {}
        all_funcs.update(self.analysis['methods'])
        all_funcs.update(self.analysis['functions'])
        
        for name, occurrences in all_funcs.items():
            # Normalize name for comparison
            normalized = re.sub(r'[^a-z]', '', name.lower())
            
            # Group by normalized name
            if len(normalized) > 5:  # Only meaningful names
                similar_functions[normalized].append({
                    'original_name': name,
                    'occurrences': len(occurrences),
                    'files': [occ['file'] for occ in occurrences]
                })
        
        # Find groups with multiple entries (potential duplicates)
        for normalized, group in similar_functions.items():
            if len(group) > 1:
                self.analysis['naming_patterns']['duplicate_functionality'].append({
                    'normalized_name': normalized,
                    'functions': group,
                    'total_occurrences': sum(func['occurrences'] for func in group)
                })
    
    def _generate_recommendations(self):
        """Generate consolidation recommendations"""
        
        recommendations = {
            'class_consolidation': [],
            'method_standardization': [],
            'naming_improvements': [],
            'code_deduplication': []
        }
        
        # Class consolidation recommendations
        ciaf_classes = [name for name in self.analysis['classes'].keys() 
                       if 'ciaf' in name.lower() or 'wrapper' in name.lower()]
        
        if len(ciaf_classes) > 3:
            recommendations['class_consolidation'].append({
                'issue': f'Found {len(ciaf_classes)} CIAF-related classes',
                'classes': ciaf_classes,
                'suggestion': 'Consider consolidating into fewer base classes with inheritance'
            })
        
        # Method standardization recommendations
        for inconsistency in self.analysis['naming_patterns']['inconsistent_naming']:
            if len(inconsistency['variants']) > 2:
                recommendations['method_standardization'].append({
                    'concept': inconsistency['base_concept'],
                    'current_variants': inconsistency['variants'],
                    'suggestion': f'Standardize to single naming pattern for {inconsistency["base_concept"]} operations'
                })
        
        # Naming improvements
        for verbose in self.analysis['naming_patterns']['verbose_names']:
            if verbose['length'] > 40:
                recommendations['naming_improvements'].append({
                    'current_name': verbose['name'],
                    'length': verbose['length'],
                    'suggestion': f'Shorten {verbose["type"]} name while preserving meaning'
                })
        
        # Code deduplication
        for duplicate in self.analysis['naming_patterns']['duplicate_functionality']:
            if duplicate['total_occurrences'] > 5:
                recommendations['code_deduplication'].append({
                    'functionality': duplicate['normalized_name'],
                    'functions': [f['original_name'] for f in duplicate['functions']],
                    'suggestion': 'Consider creating single utility function for this functionality'
                })
        
        self.analysis['recommendations'] = recommendations
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive analysis report"""
        
        report = []
        report.append("🔍 CIAF FRAMEWORK COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Summary statistics
        report.append("📊 CODEBASE OVERVIEW")
        report.append("-" * 30)
        report.append(f"Total files analyzed: {len(self.processed_files)}")
        report.append(f"Total classes: {len(self.analysis['classes'])}")
        report.append(f"Total methods: {len(self.analysis['methods'])}")
        report.append(f"Total functions: {len(self.analysis['functions'])}")
        report.append(f"Total variables: {len(self.analysis['variables'])}")
        report.append(f"Total import statements: {len(self.analysis['imports'])}")
        report.append("")
        
        # Top files by complexity
        report.append("📁 FILES BY COMPLEXITY (Lines of Code)")
        report.append("-" * 30)
        sorted_files = sorted(self.analysis['file_summary'].items(), 
                            key=lambda x: x[1]['lines_of_code'], reverse=True)
        
        for i, (file_path, info) in enumerate(sorted_files[:10]):
            classes = len(info['classes'])
            methods = len(info['methods'])
            functions = len(info['functions'])
            report.append(f"{i+1:2d}. {file_path}")
            report.append(f"    Lines: {info['lines_of_code']}, Classes: {classes}, Methods: {methods}, Functions: {functions}")
        
        report.append("")
        
        # Class analysis
        report.append("🏗️ CLASS ANALYSIS")
        report.append("-" * 30)
        
        # Most complex classes (by number of methods)
        class_complexity = {}
        for class_name, occurrences in self.analysis['classes'].items():
            total_methods = sum(len(occ['methods']) for occ in occurrences)
            class_complexity[class_name] = total_methods
        
        sorted_classes = sorted(class_complexity.items(), key=lambda x: x[1], reverse=True)
        
        report.append("Top 10 Classes by Method Count:")
        for i, (class_name, method_count) in enumerate(sorted_classes[:10]):
            files = [occ['file'] for occ in self.analysis['classes'][class_name]]
            report.append(f"{i+1:2d}. {class_name} ({method_count} methods)")
            report.append(f"    Files: {', '.join(set(files))}")
        
        report.append("")
        
        # Method/Function analysis
        report.append("⚙️ METHOD & FUNCTION ANALYSIS")
        report.append("-" * 30)
        
        # Most common method names
        method_counts = {name: len(occurrences) for name, occurrences in self.analysis['methods'].items()}
        sorted_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
        
        report.append("Most Frequently Used Method Names:")
        for i, (method_name, count) in enumerate(sorted_methods[:10]):
            report.append(f"{i+1:2d}. {method_name} ({count} occurrences)")
        
        report.append("")
        
        # Naming pattern issues
        report.append("⚠️ NAMING PATTERN ISSUES")
        report.append("-" * 30)
        
        verbose_names = self.analysis['naming_patterns']['verbose_names']
        if verbose_names:
            report.append(f"Found {len(verbose_names)} verbose names (>30 characters):")
            for item in verbose_names[:10]:  # Top 10
                report.append(f"  • {item['type']}: {item['name']} ({item['length']} chars)")
                report.append(f"    Files: {', '.join(item['files'])}")
            report.append("")
        
        inconsistent_naming = self.analysis['naming_patterns']['inconsistent_naming']
        if inconsistent_naming:
            report.append(f"Found {len(inconsistent_naming)} inconsistent naming patterns:")
            for item in inconsistent_naming[:5]:  # Top 5
                report.append(f"  • Concept: {item['base_concept']}")
                report.append(f"    Variants: {', '.join(item['variants'])}")
                report.append(f"    Patterns: {', '.join(item['patterns'])}")
            report.append("")
        
        duplicate_functionality = self.analysis['naming_patterns']['duplicate_functionality']
        if duplicate_functionality:
            report.append(f"Found {len(duplicate_functionality)} potential duplicate functionalities:")
            for item in duplicate_functionality[:5]:  # Top 5
                report.append(f"  • Functionality: {item['normalized_name']}")
                report.append(f"    Functions: {[f['original_name'] for f in item['functions']]}")
                report.append(f"    Total occurrences: {item['total_occurrences']}")
            report.append("")
        
        # Recommendations
        if 'recommendations' in self.analysis:
            report.append("💡 CONSOLIDATION RECOMMENDATIONS")
            report.append("-" * 30)
            
            recommendations = self.analysis['recommendations']
            
            if recommendations['class_consolidation']:
                report.append("Class Consolidation:")
                for rec in recommendations['class_consolidation']:
                    report.append(f"  • {rec['issue']}")
                    report.append(f"    Classes: {', '.join(rec['classes'])}")
                    report.append(f"    Suggestion: {rec['suggestion']}")
                report.append("")
            
            if recommendations['method_standardization']:
                report.append("Method Standardization:")
                for rec in recommendations['method_standardization'][:5]:
                    report.append(f"  • Concept: {rec['concept']}")
                    report.append(f"    Current variants: {', '.join(rec['current_variants'])}")
                    report.append(f"    Suggestion: {rec['suggestion']}")
                report.append("")
            
            if recommendations['naming_improvements']:
                report.append("Naming Improvements:")
                for rec in recommendations['naming_improvements'][:5]:
                    report.append(f"  • {rec['current_name']} ({rec['length']} chars)")
                    report.append(f"    Suggestion: {rec['suggestion']}")
                report.append("")
            
            if recommendations['code_deduplication']:
                report.append("Code Deduplication:")
                for rec in recommendations['code_deduplication'][:5]:
                    report.append(f"  • Functionality: {rec['functionality']}")
                    report.append(f"    Functions: {', '.join(rec['functions'])}")
                    report.append(f"    Suggestion: {rec['suggestion']}")
                report.append("")
        
        # CIAF-specific analysis
        report.append("🎯 CIAF-SPECIFIC ANALYSIS")
        report.append("-" * 30)
        
        ciaf_related = self._analyze_ciaf_specific()
        
        report.append("CIAF Models & Wrappers:")
        for category, items in ciaf_related.items():
            if items:
                report.append(f"  {category.replace('_', ' ').title()}:")
                for item in items:
                    report.append(f"    • {item}")
        
        report.append("")
        report.append("=" * 70)
        report.append("🎯 ANALYSIS COMPLETE")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_file}")
        
        return report_text
    
    def _analyze_ciaf_specific(self) -> Dict[str, List[str]]:
        """Analyze CIAF-specific classes and functions"""
        
        ciaf_analysis = {
            'model_classes': [],
            'wrapper_classes': [],
            'utility_classes': [],
            'regressor_classes': [],
            'data_utilities': [],
            'error_utilities': [],
            'lcm_components': []
        }
        
        for class_name in self.analysis['classes'].keys():
            name_lower = class_name.lower()
            
            if 'regressor' in name_lower:
                ciaf_analysis['regressor_classes'].append(class_name)
            elif 'wrapper' in name_lower:
                ciaf_analysis['wrapper_classes'].append(class_name)
            elif 'model' in name_lower and 'ciaf' in name_lower:
                ciaf_analysis['model_classes'].append(class_name)
            elif any(util in name_lower for util in ['utils', 'utility', 'helper']):
                ciaf_analysis['utility_classes'].append(class_name)
            elif 'lcm' in name_lower:
                ciaf_analysis['lcm_components'].append(class_name)
        
        for func_name in self.analysis['functions'].keys():
            name_lower = func_name.lower()
            
            if any(data_term in name_lower for data_term in ['data', 'format', 'convert']):
                ciaf_analysis['data_utilities'].append(func_name)
            elif any(error_term in name_lower for error_term in ['error', 'exception', 'translate']):
                ciaf_analysis['error_utilities'].append(func_name)
        
        return ciaf_analysis
    
    def export_json(self, output_file: str):
        """Export analysis results as JSON"""
        
        # Convert defaultdicts to regular dicts for JSON serialization
        export_data = {}
        for key, value in self.analysis.items():
            if isinstance(value, defaultdict):
                export_data[key] = dict(value)
            else:
                export_data[key] = value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📊 Analysis data exported to: {output_file}")


def main():
    """Main analysis function"""
    
    # Set up paths
    current_dir = Path(__file__).parent
    root_path = current_dir.parent.parent  # Go up to PYPI root
    
    print(f"🔍 Analyzing CIAF codebase at: {root_path}")
    print()
    
    # Create analyzer
    analyzer = CIAFCodeAnalyzer(str(root_path))
    
    # Run analysis
    results = analyzer.analyze_codebase()
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save detailed outputs
    output_dir = current_dir / "analysis_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Save text report
    report_file = output_dir / "ciaf_framework_analysis_report.txt"
    analyzer.generate_report(str(report_file))
    
    # Save JSON data
    json_file = output_dir / "ciaf_framework_analysis_data.json"
    analyzer.export_json(str(json_file))
    
    print()
    print("📁 Analysis outputs saved:")
    print(f"  📄 Text Report: {report_file}")
    print(f"  📊 JSON Data: {json_file}")
    
    # Generate consolidation plan
    print()
    print("🎯 CONSOLIDATION PLAN RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = results.get('recommendations', {})
    
    if recommendations:
        print("Priority Actions:")
        print("1. Create unified base classes for CIAF models")
        print("2. Standardize method naming across all components")
        print("3. Consolidate utility functions into modules")
        print("4. Remove duplicate functionality")
        print("5. Implement consistent error handling")
        print()
        
        print("Suggested New Framework Structure:")
        print("├── ciaf/")
        print("│   ├── core/")
        print("│   │   ├── base_model.py        # Unified base class")
        print("│   │   ├── model_wrapper.py     # Single wrapper class")
        print("│   │   └── framework.py         # Main CIAF framework")
        print("│   ├── utils/")
        print("│   │   ├── data_utils.py        # Data handling utilities")
        print("│   │   ├── wrapper_utils.py     # Wrapper utilities")
        print("│   │   └── error_utils.py       # Error handling utilities")
        print("│   ├── models/")
        print("│   │   ├── regression/")
        print("│   │   │   ├── linear.py        # CIAFLinearRegressor")
        print("│   │   │   ├── advanced.py      # CIAFAdvancedRegressor")
        print("│   │   │   └── ensemble.py      # CIAFEnsembleRegressor")
        print("│   │   └── classification/      # Future expansion")
        print("│   └── lcm/                     # LCM components")
        print("│       ├── inference.py")
        print("│       ├── provenance.py")
        print("│       └── storage.py")
        print()
        
    else:
        print("No specific recommendations generated.")
    
    return results


if __name__ == "__main__":
    main()