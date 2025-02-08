# Any Visualizer Technical Implementation

### 1. System Architecture

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   Frontend  │       │   Backend   │       │  LLM API    │       │   Redis     │
│  (React)    ◄───────► (FastAPI)   ◄───────► (Gemini)    │       │  Cache      │
└──────┬──────┘       └──────┬──────┘       └─────────────┘       └─────────────┘
       │                     │                      │                     │
┌──────▼──────┐       ┌─────▼──────┐       ┌──────▼──────┐     ┌───────▼───────┐
│   Browser   │       │   Pandas   │       │ Validation  │     │  Background   │
│ File Upload │       │ Processing │       │   Layer    │     │    Tasks      │
└─────────────┘       └───────────┘       └───────────┘      └───────────────┘
```

### 2. Project Structure

```
any-visualizer/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   ├── hooks/
│   │   └── utils/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── core/
│   │   └── utils/
└── docker/
```

### 3. Frontend Implementation

#### Key Components:

1. **FileUploadComponent**:

```typescript
interface FileUploadProps {
  maxFiles?: number;
  maxSize?: number;
  onUpload: (files: File[]) => Promise<void>;
}

const FileUpload: React.FC<FileUploadProps> = ({
  maxFiles = 3,
  maxSize = 10 * 1024 * 1024, // 10MB
  onUpload
}) => {
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    // Validation
    if (acceptedFiles.length > maxFiles) {
      throw new Error(`Maximum ${maxFiles} files allowed`);
    }

    // Size validation
    const oversizedFiles = acceptedFiles.filter(file => file.size > maxSize);
    if (oversizedFiles.length > 0) {
      throw new Error('Files exceed size limit');
    }

    // Type validation
    const nonCsvFiles = acceptedFiles.filter(
      file => !file.type.includes('csv')
    );
    if (nonCsvFiles.length > 0) {
      throw new Error('Only CSV files are allowed');
    }

    await onUpload(acceptedFiles);
  }, [maxFiles, maxSize, onUpload]);

  return (
    <Dropzone onDrop={onDrop}>
      {({getRootProps, getInputProps}) => (
        <div {...getRootProps()}>
          <input {...getInputProps()} />
          <p>Drag & drop CSV files here</p>
        </div>
      )}
    </Dropzone>
  );
};
```

2. **DataPreviewTable**:

```typescript
interface DataPreviewProps {
  data: {
    filename: string;
    preview: Array<Record<string, any>>;
    stats: Record<string, any>;
  }[];
}

const DataPreview: React.FC<DataPreviewProps> = ({ data }) => {
  return (
    <div>
      {data.map(file => (
        <div key={file.filename}>
          <h3>{file.filename}</h3>
          <Table data={file.preview} />
          <StatsDisplay stats={file.stats} />
        </div>
      ))}
    </div>
  );
};
```

3. **VisualizationContainer**:

```typescript
const VisualizationContainer: React.FC = () => {
  const [selectedViz, setSelectedViz] = useState<string | null>(null);
  const [customizations, setCustomizations] = useState({});
  const { data, isLoading, error } = useVisualization(selectedViz, customizations);

  // Fallback handling
  if (error) {
    return <FallbackVisualization data={data} error={error} />;
  }

  return (
    <div>
      <PlotlyRenderer data={data} />
      <CustomizationPanel
        onUpdate={setCustomizations}
        availableOptions={data?.customizationOptions}
      />
    </div>
  );
};
```

### 4. Backend Implementation

#### Core Services:

1. **Enhanced DataProcessor**:

```python
class DataProcessor:
    def __init__(self):
        self.cache = RedisCache()

    async def process_files(self, files: List[UploadFile]) -> Dict:
        try:
            # Process files with chunking for large files
            dfs = {}
            for file in files:
                df = await self._read_csv_chunked(file)
                dfs[file.filename] = df

            # Generate comprehensive summary
            summary = self._generate_summary(dfs)

            # Detect relationships
            relationships = self._detect_relationships(dfs)

            # Cache results
            await self.cache.set_summary(summary)

            return {
                'summary': summary,
                'relationships': relationships,
                'schema': self._extract_schema(dfs)
            }

        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            raise DataProcessingError(str(e))

    async def _read_csv_chunked(self, file: UploadFile, chunk_size=10000):
        chunks = []
        try:
            while chunk := await file.read(chunk_size):
                chunk_df = pd.read_csv(BytesIO(chunk))
                chunks.append(chunk_df)
            return pd.concat(chunks)
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            raise CSVReadError(str(e))
```

2. **Enhanced LLM Service**:

```python
class LLMService:
    def __init__(self):
        self.model = GenAI()
        self.validator = LLMOutputValidator()
        self.cache = RedisCache()

    async def generate_visualization(
        self,
        data_summary: Dict,
        retries: int = 3
    ) -> Dict:
        # Check cache first
        cache_key = self._generate_cache_key(data_summary)
        if cached := await self.cache.get(cache_key):
            return cached

        for attempt in range(retries):
            try:
                # Generate visualization code
                prompt = self._create_prompt(data_summary)
                response = await self.model.generate(prompt)

                # Validate response
                code = self.validator.extract_python_code(response)
                if not self.validator.is_valid(code):
                    raise InvalidCodeError("Generated code failed validation")

                result = {
                    'code': code,
                    'explanation': self._extract_explanation(response),
                    'metadata': self._extract_metadata(response)
                }

                # Cache successful result
                await self.cache.set(cache_key, result)
                return result

            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}): {str(e)}")
                if attempt == retries - 1:
                    return self._get_fallback_visualization(data_summary)

    def _create_prompt(self, data_summary: Dict) -> str:
        return f"""
        Given these dataset summaries:
        {json.dumps(data_summary, indent=2)}

        Generate visualization code that:
        1. Uses Plotly Express
        2. Handles missing data appropriately
        3. Includes proper axis labels and title
        4. Uses appropriate color schemes
        5. Implements basic interactivity

        Consider:
        - Data types of each column
        - Statistical distribution of values
        - Potential relationships between datasets
        - Best practices for the chosen visualization type

        Return:
        1. Python code for the visualization
        2. Explanation of visualization choices
        3. Alternative visualization suggestions
        """

    def _get_fallback_visualization(self, data_summary: Dict) -> Dict:
        """Generate a safe fallback visualization when LLM fails"""
        # Analyze data to determine appropriate visualization
        viz_type = self._determine_viz_type(data_summary)

        # Get predefined template
        template = self.fallback_templates[viz_type]

        # Fill template with data
        code = template.format(
            x_col=self._get_best_x_column(data_summary),
            y_col=self._get_best_y_column(data_summary)
        )

        return {
            'code': code,
            'is_fallback': True,
            'reason': 'LLM generation failed'
        }
```

3. **Visualization Service**:

```python
class VisualizationService:
    def __init__(self):
        self.llm_service = LLMService()
        self.executor = CodeExecutor()
        self.cache = RedisCache()

    async def create_visualization(
        self,
        data: Dict,
        customization: Optional[Dict] = None
    ) -> Dict:
        try:
            # Get visualization code
            viz_code = await self.llm_service.generate_visualization(data)

            # Apply customizations if provided
            if customization:
                viz_code = self._apply_customizations(viz_code, customization)

            # Execute code
            result = await self.executor.execute(viz_code)

            # Convert to appropriate format
            return self._format_result(result)

        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return await self._handle_visualization_error(e, data)

    async def _handle_visualization_error(
        self,
        error: Exception,
        data: Dict
    ) -> Dict:
        """Handle visualization errors with appropriate fallbacks"""
        if isinstance(error, LLMError):
            # Use fallback visualization
            return await self._generate_fallback(data)
        elif isinstance(error, ExecutionError):
            # Try simplified version
            return await self._generate_simplified(data)
        else:
            # Resort to basic visualization
            return self._generate_basic_viz(data)
```

### 5. Error Handling and Fallbacks

1. **Hierarchical Fallback System**:

```python
class FallbackSystem:
    def __init__(self):
        self.fallbacks = [
            self._try_simplified_version,
            self._try_basic_visualization,
            self._try_data_table,
            self._try_error_message
        ]

    async def handle_error(
        self,
        error: Exception,
        data: Dict,
        context: Dict
    ) -> Dict:
        for fallback in self.fallbacks:
            try:
                result = await fallback(data, context)
                if result:
                    return result
            except Exception as e:
                continue

        return self._ultimate_fallback()

    async def _try_simplified_version(self, data: Dict, context: Dict):
        """Attempt a simplified version of the original visualization"""
        # Remove complex features but keep basic visualization
        pass

    async def _try_basic_visualization(self, data: Dict, context: Dict):
        """Fall back to a very basic visualization"""
        # Use simple scatter or bar chart
        pass

    async def _try_data_table(self, data: Dict, context: Dict):
        """Display data in tabular format"""
        # When visualization fails, show data table
        pass
```

### 6. Security Considerations

1. **Code Execution Safety**:

```python
class CodeExecutor:
    def __init__(self):
        self.allowed_modules = {'pandas', 'plotly', 'numpy'}
        self.timeout = 10  # seconds

    async def execute(self, code: str) -> Dict:
        # Validate code
        self._validate_code(code)

        # Set up sandbox environment
        globals_dict = self._create_safe_globals()

        try:
            # Execute with timeout
            with timeout(self.timeout):
                exec(code, globals_dict)

            # Extract and validate result
            return self._extract_result(globals_dict)

        except Exception as e:
            raise ExecutionError(str(e))

    def _validate_code(self, code: str):
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name not in self.allowed_modules:
                        raise SecurityError(f"Unauthorized import: {name.name}")
```

### 7. Cache Strategy

```python
class CacheStrategy:
    def __init__(self):
        self.redis = Redis()

    async def get_or_generate(
        self,
        key: str,
        generator: Callable,
        ttl: int = 3600
    ):
        # Try cache first
        cached = await self.redis.get(key)
        if cached:
            return cached

        # Generate new result
        result = await generator()

        # Cache with TTL
        await self.redis.set(key, result, ex=ttl)

        return result
```

### 8. Performance Optimizations

1. **Chunked Processing**:

```python
class ChunkedProcessor:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size

    async def process_large_file(self, file: UploadFile):
        chunks = []
        async for chunk in self._read_chunks(file):
            processed = await self._process_chunk(chunk)
            chunks.append(processed)

        return self._combine_chunks(chunks)
```

2. **Background Tasks**:

```python
class BackgroundTaskManager:
    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.workers = []

    async def add_task(self, task: Callable):
        await self.task_queue.put(task)

    async def process_tasks(self):
        while True:
            task = await self.task_queue.get()
            try:
                await task()
            except Exception as e:
                logger.error(f"Background task error: {str(e)}")
            finally:
                self.task_queue.task_done()
```

### 9. Testing Strategy

1. **Test Categories**:

- Unit tests for individual components
- Integration tests for service interactions
- End-to-end tests for complete flows
- Performance tests for optimization
- Security tests for code execution
- LLM response validation tests

2. **Test Implementation**:

````python
class TestSuite:
    def __init__(self):
        self.test_cases = []

    def add_test(self, test_case: TestCase):
        self.test_cases.append(test_case)

    async def run_tests(self):
        results = []
        for test in self.test_cases:
            try:
                result = await test.run()
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    success=False,
                    error=str(e),
                    test_name=test.name
                ))

        return TestSuite.summarize_results(results)

class TestResult:
    def __init__(self, success: bool, test_name: str, error: Optional[str] = None):
        self.success = success
        self.test_name = test_name
        self.error = error
        self.timestamp = datetime.now()

class VisualizationTests:
    @pytest.mark.asyncio
    async def test_llm_visualization_generation(self):
        """Test LLM visualization code generation"""
        test_data = {
            'columns': ['date', 'value'],
            'sample': pd.DataFrame({
                'date': ['2024-01-01', '2024-01-02'],
                'value': [100, 200]
            })
        }

        llm_service = LLMService()
        result = await llm_service.generate_visualization(test_data)

        assert 'code' in result
        assert 'explanation' in result
        assert self._is_valid_python_code(result['code'])

    @pytest.mark.asyncio
    async def test_fallback_system(self):
        """Test fallback visualization generation"""
        test_data = {
            'error': 'LLM generation failed',
            'data': pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})
        }

        fallback = FallbackSystem()
        result = await fallback.handle_error(
            Exception("Test error"),
            test_data,
            {'attempt': 1}
        )

        assert result is not None
        assert 'visualization' in result
        assert result['is_fallback'] is True

class PerformanceTests:
    @pytest.mark.asyncio
    async def test_large_file_processing(self):
        """Test processing of large CSV files"""
        large_file = generate_large_test_file(size_mb=100)
        processor = ChunkedProcessor()

        start_time = time.time()
        result = await processor.process_large_file(large_file)
        processing_time = time.time() - start_time

        assert processing_time < 30  # Should process within 30 seconds
        assert result is not None
        assert not result.empty

class SecurityTests:
    @pytest.mark.asyncio
    async def test_code_injection_prevention(self):
        """Test prevention of malicious code execution"""
        malicious_code = """
        import os
        os.system('rm -rf /')
        """

        executor = CodeExecutor()
        with pytest.raises(SecurityError):
            await executor.execute(malicious_code)

    @pytest.mark.asyncio
    async def test_file_upload_validation(self):
        """Test file upload security"""
        with open('test_malicious.csv', 'wb') as f:
            f.write(b'\x00\x00\x00')  # Binary content

        with pytest.raises(MaliciousFileError):
            await validate_csv_file('test_malicious.csv')

### 10. Deployment and DevOps

1. **Docker Optimization**:
```dockerfile
# Frontend Dockerfile with multi-stage build
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

# Backend Dockerfile with optimization
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
````

2. **Kubernetes Deployment**:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: any-visualizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: any-visualizer
  template:
    metadata:
      labels:
        app: any-visualizer
    spec:
      containers:
        - name: backend
          image: any-visualizer-backend:latest
          resources:
            limits:
              cpu: "1"
              memory: "1Gi"
            requests:
              cpu: "500m"
              memory: "512Mi"
          env:
            - name: GEMINI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-keys
                  key: gemini
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
```

3. **CI/CD Pipeline**:

```yaml
# GitHub Actions workflow
name: CI/CD
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.9"
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Deployment steps here
```

### 11. Monitoring and Observability

1. **Application Monitoring**:

```python
from prometheus_client import Counter, Histogram, start_http_server

class MetricsCollector:
    def __init__(self):
        self.visualization_requests = Counter(
            'visualization_requests_total',
            'Total visualization requests'
        )
        self.llm_latency = Histogram(
            'llm_request_latency_seconds',
            'LLM request latency'
        )
        self.visualization_errors = Counter(
            'visualization_errors_total',
            'Total visualization errors',
            ['error_type']
        )

    async def record_request(self):
        self.visualization_requests.inc()

    async def record_llm_latency(self, start_time):
        self.llm_latency.observe(time.time() - start_time)

    async def record_error(self, error_type):
        self.visualization_errors.labels(error_type=error_type).inc()
```

2. **Logging System**:

```python
import structlog

logger = structlog.get_logger()

class LoggingMiddleware:
    async def __call__(self, request, call_next):
        request_id = str(uuid.uuid4())
        logger.bind(request_id=request_id)

        logger.info(
            "request_started",
            path=request.url.path,
            method=request.method
        )

        try:
            response = await call_next(request)
            logger.info(
                "request_completed",
                status_code=response.status_code
            )
            return response
        except Exception as e:
            logger.error(
                "request_failed",
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise
```

3. **Health Checks**:

```python
class HealthCheck:
    async def check_services(self):
        results = {}

        # Check Redis
        try:
            await self.redis.ping()
            results['redis'] = 'healthy'
        except Exception as e:
            results['redis'] = f'unhealthy: {str(e)}'

        # Check LLM API
        try:
            await self.llm_service.test_connection()
            results['llm_api'] = 'healthy'
        except Exception as e:
            results['llm_api'] = f'unhealthy: {str(e)}'

        return results

    @app.get("/health")
    async def health_check():
        checker = HealthCheck()
        results = await checker.check_services()

        all_healthy = all(v == 'healthy' for v in results.values())
        return {
            'status': 'healthy' if all_healthy else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'services': results
        }
```

### 12. Scaling Considerations

1. **Horizontal Scaling**:

- Use Redis for session storage
- Implement sticky sessions
- Store visualizations in shared storage
- Use message queue for background tasks

2. **Resource Management**:

```python
class ResourceManager:
    def __init__(self):
        self.max_concurrent = 10
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    async def process_with_limits(self, func):
        async with self.semaphore:
            return await func()
```

3. **Load Balancing**:

```nginx
# nginx.conf
upstream backend {
    least_conn;  # Use least connections algorithm
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
