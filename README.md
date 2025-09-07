# CodeQ: Advanced Code Quality Analyzer

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CodeQ** es un analizador avanzado de calidad de código multi-lenguaje y explicable que calcula y reporta métricas de calidad con alta precisión y machine learning integrado.

## 🚀 Características Principales

### 📊 Métricas Avanzadas
- **Complejidad**: Ciclomática, cognitiva, Halstead (V, D, E, T, B)
- **Mantenibilidad**: Índice de Mantenibilidad con bandas A/B/C
- **Acoplamiento/Cohesión**: Fan-in/out, inestabilidad, LCOM, CBO, RFC
- **Code Smells**: 15+ patrones configurables
- **Documentación**: Cobertura de docstrings, ratio comentarios/código
- **Deuda Técnica**: Estimación SQALE-like con costos en €/hora

### 🤖 Machine Learning Avanzado
- **Modelos Ensemble**: XGBoost, LightGBM, CatBoost, Redes Neuronales
- **Predicciones**: Riesgo de bugs, pronóstico de calidad, anomalías
- **AutoML**: Entrenamiento automático con validación cruzada
- **60+ features**: Extraídas automáticamente del código

### 🌍 Soporte Multi-Lenguaje
- **Python**: Análisis completo con AST y tree-sitter
- **TypeScript/JavaScript**: Soporte completo para TSX/JSX
- **Go, Rust, Java, C#, C++, PHP, Ruby, Swift, Kotlin, Scala**: Arquitectura extensible

## 📋 Requisitos del Sistema

- **Python**: 3.11 o superior
- **RAM**: 4GB mínimo, 8GB recomendado
- **Disco**: 500MB para instalación
- **SO**: Linux, macOS, Windows (con WSL)

## 🛠️ Instalación

### Opción 1: Instalación desde código fuente

```bash
# Clonar el repositorio
git clone https://github.com/your-org/codeq.git
cd codeq

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python main.py --help
```

### Opción 2: Usando Docker

```bash
# Construir imagen
docker build -t codeq .

# Ejecutar contenedor
docker run -v $(pwd):/workspace codeq scan /workspace
```

## 📖 Uso Básico

### Escaneo Simple

```bash
# Escanear directorio actual
python main.py scan .

# Escanear proyecto específico
python main.py scan /path/to/project

# Escanear con lenguajes específicos
python main.py scan . --lang py,ts,js
```

### Reportes Avanzados

```bash
# Generar reporte HTML
python main.py scan . --report analysis.html

# Generar JSON detallado con ML
python main.py scan . --detailed-json report.json --verbose

# Generar SARIF para CI/CD
python main.py scan . --sarif results.sarif

# Combinar múltiples formatos
python main.py scan . --report analysis.html --json report.json --sarif results.sarif
```

### Opciones de Calidad

```bash
# Fallar si hay problemas críticos
python main.py scan . --fail-on 'severity>=major'

# Limitar deuda técnica
python main.py scan . --budget-hours 80

# Análisis con workers paralelos
python main.py scan . --workers 8
```

## 📊 Salida y Reportes

### Formato JSON (Machine-readable)

```json
{
  "metadata": {
    "tool": "CodeQ",
    "version": "0.1.0",
    "timestamp": "2024-01-15T10:30:00Z",
    "scan_duration_seconds": 45.2,
    "platform": {"system": "Linux", "python_version": "3.11.5"}
  },
  "summary": {
    "overall_score": 78.5,
    "quality_rating": "B",
    "risk_level": "Medium",
    "total_files": 156,
    "total_lines": 12543,
    "languages_detected": ["python", "typescript"]
  },
  "detailed_metrics": {
    "complexity_metrics": {
      "avg_cyclomatic_complexity": 8.2,
      "max_cyclomatic_complexity": 32,
      "cognitive_complexity_avg": 12.5
    },
    "maintainability": {
      "average_mi": 68.3,
      "band_distribution": {"A": 12, "B": 89, "C": 55}
    }
  },
  "detailed_findings": [
    {
      "id": "01a",
      "rule": "long-function",
      "severity": "major",
      "message": "Function exceeds 50 lines",
      "position": {
        "path": "src/main.py",
        "start_line": 45,
        "end_line": 98
      },
      "snippet": "def process_data(data):\\n    # 53 lines of code...",
      "rationale": "Large functions are hard to understand and maintain",
      "remediation_minutes": 60,
      "tags": ["maintainability", "complexity"],
      "confidence": 0.95
    }
  ],
  "ml_predictions": {
    "bug_likelihood": {
      "prediction": 0.23,
      "confidence": "HIGH",
      "explanation": "Code has moderate bug risk due to complexity patterns"
    },
    "quality_forecast": {
      "prediction": 75.2,
      "confidence": "MEDIUM",
      "explanation": "Quality expected to stabilize with current trends"
    }
  }
}
```

### Reporte HTML Interactivo

Los reportes HTML incluyen:
- 📊 Gráficos interactivos con Plotly
- 📈 Tendencias históricas
- 🎯 Recomendaciones priorizadas
- 🔍 Búsqueda y filtrado
- 📋 Resumen ejecutivo

## 🔧 Arquitectura del Sistema

```
CodeQ/
├── main.py              # Punto de entrada principal
├── requirements.txt      # Dependencias Python
├── rules/               # Configuración de reglas
│   └── defaults.yaml    # Reglas por defecto
└── src/codeq/          # Módulos principales
    ├── __init__.py      # API pública
    ├── cli.py          # Interfaz de línea de comandos
    ├── astparse.py     # Parsing AST con tree-sitter
    ├── metrics.py      # Cálculo de métricas de complejidad
    ├── smells.py       # Detección de code smells
    ├── coupling.py     # Análisis de acoplamiento
    ├── aggregate.py    # Agregación y scoring
    ├── ml_engine.py    # Motor de machine learning
    ├── report.py       # Generación de reportes
    ├── utils.py        # Utilidades comunes
    └── recommendations.py # Sistema de recomendaciones
```

### Componentes Principales

1. **CLI (cli.py)**: Interfaz de usuario, parsing de argumentos
2. **AST Parser (astparse.py)**: Análisis sintáctico con tree-sitter
3. **Metrics Engine (metrics.py)**: Cálculo de complejidad y mantenibilidad
4. **Smell Detector (smells.py)**: Identificación de patrones problemáticos
5. **ML Engine (ml_engine.py)**: Modelos de machine learning avanzados
6. **Aggregator (aggregate.py)**: Consolidación de métricas y scoring
7. **Report Generator (report.py)**: Creación de reportes en múltiples formatos

## 🤖 Machine Learning Features

### Modelos Soportados

- **XGBoost**: Gradient boosting para clasificación
- **LightGBM**: Efficient gradient boosting
- **CatBoost**: Categorical features support
- **Random Forest**: Ensemble bagging
- **Neural Networks**: Deep learning para patrones complejos
- **SVM**: Support Vector Machines para outliers

### Características de ML

```python
# 60+ features extraídas automáticamente
features = MLFeatureVector(
    # Complejidad
    cyclomatic_complexity=12,
    cognitive_complexity=15,
    halstead_volume=250,

    # Estructural
    lines_of_code=120,
    num_functions=8,
    nesting_depth_max=4,

    # Calidad
    test_coverage=75.0,
    documentation_score=60.0,
    maintainability_index=85.0,

    # Patrón de seguridad (eliminado)
    has_hardcoded_secrets=False,
    has_sql_injection_risk=False
)
```

## 🚀 Integración CI/CD

### GitHub Actions

```yaml
# .github/workflows/codeq.yml
name: Code Quality Analysis

on: [push, pull_request]

jobs:
  codeq:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run CodeQ analysis
      run: |
        python main.py scan . \\
          --detailed-json report.json \\
          --sarif results.sarif \\
          --fail-on 'severity>=major'
    - name: Upload SARIF
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: results.sarif
```

### GitLab CI

```yaml
# .gitlab-ci.yml
code_quality:
  stage: test
  script:
    - pip install -r requirements.txt
    - python main.py scan . --sarif gl-sarif.sarif
  artifacts:
    reports:
      sarif: gl-sarif.sarif
```

## 🐛 Solución de Problemas

### Problemas Comunes

**Error: "Tree-sitter setup required"**
```bash
# Instalar parsers de tree-sitter
pip install tree-sitter tree-sitter-python tree-sitter-typescript
```

**Error: "Module not found"**
```bash
# Asegurar que estamos en el directorio correcto
cd /path/to/codeq
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

**Análisis lento en proyectos grandes**
```bash
# Usar paralelización y cache
python main.py scan . --workers 8 --cache-dir .codeq_cache
```

### Logs y Debug

```bash
# Modo verbose
python main.py scan . --verbose

# Debug completo
python main.py scan . -vv
```

## 📈 Rendimiento y Escalabilidad

### Benchmarks

- **Archivos pequeños** (< 100 archivos): < 30 segundos
- **Proyectos medianos** (100-1000 archivos): < 2 minutos
- **Proyectos grandes** (> 1000 archivos): < 5 minutos con paralelización

### Optimizaciones

- **Cache LRU**: Evita re-análisis de archivos sin cambios
- **Paralelización**: Multi-core processing con ThreadPoolExecutor
- **Análisis incremental**: Solo archivos modificados (integración Git)
- **Lazy loading**: AST parsing bajo demanda

## 🤝 Soporte y Comunidad

- 📖 **Documentación**: [docs.codeq.dev](https://docs.codeq.dev)
- 💬 **Discord**: [discord.gg/codeq](https://discord.gg/codeq)
- 🐛 **Issues**: [github.com/your-org/codeq/issues](https://github.com/your-org/codeq/issues)
- 📧 **Email**: support@codeq.dev

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

**CodeQ** - Elevando la calidad del código con inteligencia artificial avanzada. 🚀✨