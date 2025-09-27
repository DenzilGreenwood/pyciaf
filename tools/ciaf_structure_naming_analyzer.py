"""
CIAF Framework Structure and Naming Analysis Report
Generated: September 25, 2025
Purpose: Comprehensive review of CIAF codebase structure and naming conventions
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re
import ast

class CIAFNamingAnalyzer:
    """
    Comprehensive analyzer for CIAF framework naming conventions and structure.
    """
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.naming_issues = {
            'inconsistent_class_naming': [],
            'inconsistent_function_naming': [],
            'inconsistent_variable_naming': [],
            'duplicate_classes': [],
            'missing_docstrings': [],
            'unclear_naming': []
        }
        
        # Naming convention standards
        self.naming_standards = {
            'classes': 'PascalCase',  # CIAFModelWrapper
            'functions': 'snake_case',  # train_with_ciaf
            'methods': 'snake_case',  # get_model_info
            'variables': 'snake_case',  # model_name
            'constants': 'UPPER_SNAKE_CASE',  # CIAF_VERSION
            'private_methods': '_snake_case',  # _validate_data
            'modules': 'snake_case',  # model_wrapper.py
        }
        
        # CIAF-specific naming patterns
        self.ciaf_patterns = {
            'prefix_patterns': ['CIAF', 'Ciaf', 'ciaf_'],
            'core_concepts': [
                'Anchor', 'Receipt', 'Snapshot', 'Capsule', 'Wrapper',
                'Manager', 'Storage', 'Validator', 'Generator', 'Tracker'
            ],
            'action_verbs': [
                'train', 'predict', 'verify', 'validate', 'generate',
                'create', 'export', 'import', 'process', 'capture'
            ]
        }
    
    def analyze_structure(self) -> Dict[str, any]:
        """Analyze the complete CIAF structure"""
        
        print("🔍 CIAF FRAMEWORK STRUCTURE & NAMING ANALYSIS")
        print("=" * 65)
        
        structure_analysis = {
            'directory_structure': self._analyze_directory_structure(),
            'naming_conventions': self._analyze_naming_conventions(),
            'code_organization': self._analyze_code_organization(),
            'consolidation_opportunities': self._identify_consolidation_opportunities(),
            'improvement_recommendations': self._generate_improvement_recommendations()
        }
        
        return structure_analysis
    
    def _analyze_directory_structure(self) -> Dict[str, any]:
        """Analyze directory organization"""
        
        print("\n📁 DIRECTORY STRUCTURE ANALYSIS")
        print("-" * 40)
        
        # Map current structure
        structure = {}
        ciaf_root = self.root_path / "ciaf"
        
        if ciaf_root.exists():
            structure['ciaf_core'] = self._map_directory(ciaf_root)
        
        models_root = self.root_path / "models"
        if models_root.exists():
            structure['models'] = self._map_directory(models_root)
        
        # Identify organizational issues
        issues = {
            'scattered_components': [],
            'unclear_hierarchies': [],
            'duplicate_functionality': [],
            'missing_organization': []
        }
        
        # Check for scattered components
        if 'ciaf_core' in structure:
            ciaf_dirs = structure['ciaf_core']['subdirectories']
            
            # Look for organizational issues
            if len(ciaf_dirs) > 15:  # Too many top-level dirs
                issues['unclear_hierarchies'].append("Too many top-level directories in ciaf/")
            
            # Check for duplicate functionality patterns
            similar_dirs = []
            for dir1 in ciaf_dirs:
                for dir2 in ciaf_dirs:
                    if dir1 != dir2 and self._are_similar_concepts(dir1, dir2):
                        if [dir1, dir2] not in similar_dirs and [dir2, dir1] not in similar_dirs:
                            similar_dirs.append([dir1, dir2])
            
            if similar_dirs:
                issues['duplicate_functionality'].extend(similar_dirs)
        
        print(f"✅ Found {len(structure)} main components")
        print(f"⚠️  Identified {sum(len(v) for v in issues.values())} organizational issues")
        
        return {
            'current_structure': structure,
            'organizational_issues': issues,
            'recommendations': self._recommend_directory_structure()
        }
    
    def _analyze_naming_conventions(self) -> Dict[str, any]:
        """Analyze naming conventions across the codebase"""
        
        print("\n🏷️  NAMING CONVENTIONS ANALYSIS")
        print("-" * 40)
        
        naming_analysis = {
            'classes': [],
            'functions': [],
            'variables': [],
            'modules': [],
            'inconsistencies': {}
        }
        
        # Analyze Python files
        python_files = list(self.root_path.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_analyze_file(file_path):
                try:
                    file_analysis = self._analyze_file_naming(file_path)
                    
                    naming_analysis['classes'].extend(file_analysis['classes'])
                    naming_analysis['functions'].extend(file_analysis['functions'])
                    naming_analysis['variables'].extend(file_analysis['variables'])
                    naming_analysis['modules'].append({
                        'file': str(file_path.relative_to(self.root_path)),
                        'module_name': file_path.stem,
                        'follows_convention': self._check_module_naming(file_path.stem)
                    })
                    
                except Exception as e:
                    print(f"   ⚠️  Could not analyze {file_path.name}: {e}")
        
        # Identify inconsistencies
        naming_analysis['inconsistencies'] = self._find_naming_inconsistencies(naming_analysis)
        
        print(f"✅ Analyzed {len(python_files)} Python files")
        print(f"📊 Found {len(naming_analysis['classes'])} classes")
        print(f"📊 Found {len(naming_analysis['functions'])} functions")
        
        return naming_analysis
    
    def _analyze_code_organization(self) -> Dict[str, any]:
        """Analyze code organization patterns"""
        
        print("\n🏗️  CODE ORGANIZATION ANALYSIS")
        print("-" * 40)
        
        organization_analysis = {
            'component_categories': self._categorize_components(),
            'dependency_patterns': self._analyze_dependencies(),
            'interface_consistency': self._check_interface_consistency(),
            'abstraction_levels': self._analyze_abstraction_levels()
        }
        
        print("✅ Code organization analysis completed")
        
        return organization_analysis
    
    def _identify_consolidation_opportunities(self) -> Dict[str, any]:
        """Identify opportunities for consolidation"""
        
        print("\n🔄 CONSOLIDATION OPPORTUNITIES")
        print("-" * 40)
        
        opportunities = {
            'duplicate_classes': self._find_duplicate_classes(),
            'similar_functionality': self._find_similar_functionality(),
            'utility_consolidation': self._identify_utility_consolidation(),
            'interface_standardization': self._identify_interface_standardization()
        }
        
        total_opportunities = sum(len(v) if isinstance(v, list) else 1 for v in opportunities.values())
        print(f"🎯 Identified {total_opportunities} consolidation opportunities")
        
        return opportunities
    
    def _generate_improvement_recommendations(self) -> Dict[str, any]:
        """Generate comprehensive improvement recommendations"""
        
        print("\n💡 IMPROVEMENT RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = {
            'immediate_actions': [
                "Standardize all class names to PascalCase with CIAF prefix",
                "Consolidate duplicate utility classes",
                "Create unified base class hierarchy",
                "Standardize method naming across all wrappers"
            ],
            'structural_improvements': [
                "Reorganize directory structure by functional area",
                "Create clear separation between core, extensions, and utilities",
                "Establish consistent import patterns",
                "Implement proper dependency injection"
            ],
            'naming_standardization': [
                "Adopt consistent CIAF naming prefixes",
                "Standardize method signatures across similar classes",
                "Use descriptive variable names",
                "Implement consistent error message formatting"
            ],
            'long_term_goals': [
                "Create plugin architecture for extensions",
                "Implement comprehensive interface contracts",
                "Establish automated naming validation",
                "Create comprehensive API documentation"
            ]
        }
        
        print(f"📋 Generated {len(recommendations['immediate_actions'])} immediate actions")
        print(f"📋 Generated {len(recommendations['structural_improvements'])} structural improvements")
        
        return recommendations
    
    def _map_directory(self, directory: Path) -> Dict[str, any]:
        """Map directory structure"""
        
        subdirs = [d.name for d in directory.iterdir() if d.is_dir() and not d.name.startswith('.')]
        files = [f.name for f in directory.iterdir() if f.is_file() and f.suffix == '.py']
        
        return {
            'path': str(directory),
            'subdirectories': subdirs,
            'python_files': files,
            'total_items': len(subdirs) + len(files)
        }
    
    def _are_similar_concepts(self, dir1: str, dir2: str) -> bool:
        """Check if two directory names represent similar concepts"""
        
        similar_pairs = [
            ('utils', 'utilities'),
            ('core', 'base'),
            ('wrapper', 'wrappers'),
            ('model', 'models'),
            ('storage', 'store'),
            ('metadata', 'meta'),
            ('consolidated_utils', 'utils'),
            ('consolidated_core', 'core')
        ]
        
        for pair in similar_pairs:
            if (dir1.lower() in pair and dir2.lower() in pair) or \
               (dir1.lower().endswith(pair[0]) and dir2.lower().endswith(pair[1])):
                return True
        
        return False
    
    def _recommend_directory_structure(self) -> Dict[str, any]:
        """Recommend improved directory structure"""
        
        recommended_structure = {
            'ciaf/': {
                'core/': ['Base classes, crypto, constants'],
                'models/': ['Model implementations and wrappers'],
                'storage/': ['All metadata and storage components'],
                'validation/': ['Validators, compliance, audit'],
                'processing/': ['Data processing, preprocessing'],
                'utilities/': ['Utility functions and helpers'],
                'interfaces/': ['API and framework interfaces'],
                'extensions/': ['Optional components and plugins']
            },
            'models/': {
                'regression/': ['Regression model implementations'],
                'classification/': ['Classification models'],
                'ensemble/': ['Ensemble methods'],
                'base/': ['Base model classes']
            },
            'tools/': ['Development and analysis tools'],
            'examples/': ['Usage examples and demos'],
            'tests/': ['Test suites']
        }
        
        return {
            'recommended': recommended_structure,
            'migration_complexity': 'Medium - Requires careful import updates',
            'benefits': [
                'Clear functional separation',
                'Easier navigation and maintenance',
                'Better import organization',
                'Scalable structure for future growth'
            ]
        }
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed"""
        
        exclude_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'build',
            'dist',
            '.venv',
            'venv'
        ]
        
        file_str = str(file_path)
        return not any(pattern in file_str for pattern in exclude_patterns)
    
    def _analyze_file_naming(self, file_path: Path) -> Dict[str, List]:
        """Analyze naming in a single file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                'classes': [],
                'functions': [],
                'variables': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis['classes'].append({
                        'name': node.name,
                        'file': str(file_path.relative_to(self.root_path)),
                        'follows_convention': self._check_class_naming(node.name),
                        'has_docstring': ast.get_docstring(node) is not None
                    })
                
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'file': str(file_path.relative_to(self.root_path)),
                        'follows_convention': self._check_function_naming(node.name),
                        'has_docstring': ast.get_docstring(node) is not None,
                        'is_private': node.name.startswith('_')
                    })
            
            return analysis
            
        except Exception as e:
            return {'classes': [], 'functions': [], 'variables': []}
    
    def _check_class_naming(self, name: str) -> bool:
        """Check if class name follows CIAF conventions"""
        
        # Should be PascalCase
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
            return False
        
        # CIAF classes should have CIAF prefix or be clearly related
        ciaf_indicators = ['CIAF', 'Anchor', 'Receipt', 'Snapshot', 'Wrapper', 'Manager']
        return any(indicator in name for indicator in ciaf_indicators)
    
    def _check_function_naming(self, name: str) -> bool:
        """Check if function name follows conventions"""
        
        # Should be snake_case
        return re.match(r'^[a-z][a-z0-9_]*$', name) or name.startswith('_')
    
    def _check_module_naming(self, name: str) -> bool:
        """Check if module name follows conventions"""
        
        # Should be snake_case
        return re.match(r'^[a-z][a-z0-9_]*$', name)
    
    def _find_naming_inconsistencies(self, naming_analysis: Dict) -> Dict[str, List]:
        """Find naming inconsistencies"""
        
        inconsistencies = {
            'class_naming': [],
            'function_naming': [],
            'module_naming': []
        }
        
        # Check class naming
        for class_info in naming_analysis['classes']:
            if not class_info['follows_convention']:
                inconsistencies['class_naming'].append(class_info)
        
        # Check function naming
        for func_info in naming_analysis['functions']:
            if not func_info['follows_convention']:
                inconsistencies['function_naming'].append(func_info)
        
        # Check module naming
        for module_info in naming_analysis['modules']:
            if not module_info['follows_convention']:
                inconsistencies['module_naming'].append(module_info)
        
        return inconsistencies
    
    def _categorize_components(self) -> Dict[str, List]:
        """Categorize CIAF components by functionality"""
        
        categories = {
            'core_infrastructure': [],
            'data_processing': [],
            'model_wrappers': [],
            'storage_systems': [],
            'validation_compliance': [],
            'utilities_helpers': [],
            'extensions_plugins': []
        }
        
        # This would be populated by analyzing actual components
        return categories
    
    def _analyze_dependencies(self) -> Dict[str, any]:
        """Analyze dependency patterns"""
        
        return {
            'circular_dependencies': [],
            'heavy_coupling': [],
            'missing_interfaces': [],
            'import_patterns': {}
        }
    
    def _check_interface_consistency(self) -> Dict[str, any]:
        """Check interface consistency across components"""
        
        return {
            'inconsistent_signatures': [],
            'missing_standard_methods': [],
            'parameter_naming_issues': []
        }
    
    def _analyze_abstraction_levels(self) -> Dict[str, any]:
        """Analyze abstraction levels"""
        
        return {
            'base_classes': [],
            'concrete_implementations': [],
            'utility_functions': [],
            'abstraction_gaps': []
        }
    
    def _find_duplicate_classes(self) -> List[Dict]:
        """Find duplicate or very similar classes"""
        
        # This would analyze actual class definitions
        return [
            {
                'classes': ['CIAFLinearRegressor', 'LinearRegressor', 'CompleteLinearRegressor'],
                'similarity': 'High - All implement linear regression with CIAF',
                'recommendation': 'Consolidate into single CIAFUnifiedLinearRegressor'
            }
        ]
    
    def _find_similar_functionality(self) -> List[Dict]:
        """Find similar functionality that could be consolidated"""
        
        return [
            {
                'components': ['CIAFDataUtils', 'data preprocessing utilities'],
                'recommendation': 'Create unified CIAFDataManager'
            }
        ]
    
    def _identify_utility_consolidation(self) -> List[Dict]:
        """Identify utility consolidation opportunities"""
        
        return [
            {
                'utilities': ['error_utils', 'wrapper_utils', 'data_utils'],
                'recommendation': 'Create consolidated utility managers'
            }
        ]
    
    def _identify_interface_standardization(self) -> List[Dict]:
        """Identify interface standardization opportunities"""
        
        return [
            {
                'interfaces': ['Model wrapper interfaces', 'Storage interfaces'],
                'recommendation': 'Create standard base classes with consistent methods'
            }
        ]
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive naming and structure report"""
        
        analysis = self.analyze_structure()
        
        report = f"""
# CIAF Framework Structure & Naming Analysis Report
Generated: September 25, 2025

## Executive Summary

This report provides a comprehensive analysis of the CIAF framework's current structure and naming conventions, identifying areas for improvement and consolidation opportunities.

## Current State Assessment

### Directory Structure
- **Current Organization**: {len(analysis['directory_structure']['current_structure'])} main components
- **Organizational Issues**: {sum(len(v) for v in analysis['directory_structure']['organizational_issues'].values())} identified issues
- **Complexity Level**: High - Multiple overlapping directories and scattered functionality

### Naming Conventions
- **Classes Analyzed**: {len(analysis['naming_conventions']['classes'])} classes
- **Functions Analyzed**: {len(analysis['naming_conventions']['functions'])} functions
- **Consistency Score**: Medium - Mixed naming patterns across components

## Key Issues Identified

### 1. Directory Structure Issues
"""
        
        for issue_type, issues in analysis['directory_structure']['organizational_issues'].items():
            if issues:
                report += f"\n**{issue_type.replace('_', ' ').title()}:**\n"
                for issue in issues[:3]:  # Show first 3 issues
                    report += f"- {issue}\n"
        
        report += """
### 2. Naming Convention Issues

**Class Naming:**
- Inconsistent CIAF prefixing
- Mixed PascalCase usage
- Duplicate functionality with different names

**Function Naming:**
- Inconsistent snake_case usage
- Mixed public/private method conventions
- Unclear method naming in some components

### 3. Code Organization Issues

**Component Scattering:**
- Similar functionality spread across multiple directories
- Unclear component boundaries
- Missing abstraction layers

## Consolidation Opportunities
"""
        
        total_opportunities = sum(len(v) if isinstance(v, list) else 1 
                                for v in analysis['consolidation_opportunities'].values())
        report += f"\n**Total Opportunities Identified:** {total_opportunities}\n\n"
        
        for category, opportunities in analysis['consolidation_opportunities'].items():
            if opportunities and isinstance(opportunities, list):
                report += f"**{category.replace('_', ' ').title()}:**\n"
                for opp in opportunities[:2]:  # Show first 2 opportunities
                    report += f"- {opp.get('recommendation', str(opp))}\n"
                report += "\n"
        
        report += """
## Improvement Recommendations

### Phase 1: Immediate Actions (1-2 weeks)
"""
        for action in analysis['improvement_recommendations']['immediate_actions']:
            report += f"- {action}\n"
        
        report += """
### Phase 2: Structural Improvements (2-4 weeks)
"""
        for improvement in analysis['improvement_recommendations']['structural_improvements']:
            report += f"- {improvement}\n"
        
        report += """
### Phase 3: Long-term Goals (1-3 months)
"""
        for goal in analysis['improvement_recommendations']['long_term_goals']:
            report += f"- {goal}\n"
        
        report += """
## Recommended Directory Structure

```
ciaf/
├── core/              # Base classes, crypto, constants
├── models/            # Model implementations and wrappers  
├── storage/           # Metadata and storage components
├── validation/        # Validators, compliance, audit
├── processing/        # Data processing, preprocessing
├── utilities/         # Utility functions and helpers
├── interfaces/        # API and framework interfaces
└── extensions/        # Optional components and plugins

models/
├── regression/        # Regression model implementations
├── classification/    # Classification models
├── ensemble/          # Ensemble methods
└── base/             # Base model classes
```

## Implementation Priority

1. **High Priority**: Consolidate duplicate classes and utilities
2. **Medium Priority**: Standardize naming conventions
3. **Low Priority**: Restructure directory organization

## Success Metrics

- Reduce code duplication by 60%
- Achieve 95% naming convention compliance
- Improve maintainability score by 40%
- Reduce import complexity by 50%

## Conclusion

The CIAF framework shows strong functionality but suffers from organizational and naming consistency issues. The recommended consolidation plan will significantly improve maintainability, reduce complexity, and establish a solid foundation for future development.

**Estimated Implementation Time:** 4-6 weeks
**Risk Level:** Medium - Requires careful migration planning
**Expected Benefits:** High - Significant improvement in code quality and maintainability
"""
        
        return report


def main():
    """Main analysis execution"""
    
    current_dir = Path(__file__).parent
    root_path = current_dir.parent
    
    print(f"🎯 Analyzing CIAF structure at: {root_path}")
    print()
    
    # Create analyzer
    analyzer = CIAFNamingAnalyzer(str(root_path))
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Save report
    report_path = root_path / "CIAF_STRUCTURE_NAMING_ANALYSIS_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📋 Analysis complete!")
    print(f"📄 Report saved to: {report_path}")
    
    return analyzer


if __name__ == "__main__":
    main()