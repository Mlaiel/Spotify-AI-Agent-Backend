# Billing System Tests

## Overview

This directory contains comprehensive tests for the advanced billing system, covering all aspects of subscription management, payment processing, invoicing, analytics, and background tasks.

## Test Structure

```
tests_backend/app/billing/
├── __init__.py              # Test package initialization
├── conftest.py              # Test fixtures and configuration
├── test_models.py           # Database model tests
├── test_core.py             # Core billing engine tests
├── test_api.py              # FastAPI endpoint tests
├── test_invoices.py         # Invoice management tests
├── test_webhooks.py         # Webhook processing tests
├── test_analytics.py        # Analytics and reporting tests
└── test_tasks.py            # Background task tests
```

## Test Categories

### 🗄️ Model Tests (`test_models.py`)
- **Customer Model**: Creation, validation, relationships
- **Plan Model**: Pricing, intervals, features, usage limits
- **Subscription Model**: Lifecycle, status changes, calculations
- **Payment Model**: Processing, failures, refunds, risk scoring
- **Invoice Model**: Generation, payment tracking, overdue logic
- **PaymentMethod Model**: Card validation, expiration, security
- **Relationships**: Foreign keys, cascading, data integrity

### ⚙️ Core Engine Tests (`test_core.py`)
- **BillingEngine**: Customer/subscription management, lifecycle
- **PaymentProcessor**: Multi-provider support (Stripe, PayPal)
- **TaxCalculator**: EU VAT, US sales tax, reverse charge
- **FraudDetection**: Risk scoring, ML predictions, velocity checks
- **Integration**: End-to-end billing workflows

### 🌐 API Tests (`test_api.py`)
- **Customer Endpoints**: CRUD operations, validation
- **Plan Endpoints**: Creation, updates, deactivation
- **Subscription Endpoints**: Lifecycle management, upgrades
- **Payment Endpoints**: Processing, refunds, methods
- **Invoice Endpoints**: Generation, payment, PDF download
- **Webhook Endpoints**: Stripe/PayPal event handling
- **Analytics Endpoints**: Revenue reports, metrics
- **Error Handling**: Validation, authorization, rate limiting

### 📄 Invoice Tests (`test_invoices.py`)
- **InvoiceService**: Generation, finalization, payment tracking
- **PDFGenerator**: Multi-language PDFs, templates, attachments
- **EmailService**: Invoice delivery, reminders, confirmations
- **Dunning Management**: Automated collection workflows
- **Integration**: Complete invoice-to-payment workflows

### 🔗 Webhook Tests (`test_webhooks.py`)
- **WebhookProcessor**: Event routing, retry logic, deduplication
- **StripeWebhookHandler**: Payment intents, subscriptions, setup intents
- **PayPalWebhookHandler**: Payments, subscriptions, notifications
- **Security**: Signature verification, IP whitelisting, rate limiting
- **Monitoring**: Logging, metrics, error tracking

### 📊 Analytics Tests (`test_analytics.py`)
- **AnalyticsService**: Revenue, subscription, customer metrics
- **ReportGenerator**: Monthly reports, segmentation, exports
- **ForecastingEngine**: Revenue prediction, churn analysis, LTV
- **Performance**: Caching, query optimization, real-time data
- **Integration**: Dashboard data, scheduled reports

### 🔄 Task Tests (`test_tasks.py`)
- **BillingTaskManager**: Task scheduling, monitoring, cancellation
- **Subscription Renewals**: Automated billing cycles, dunning
- **Payment Retries**: Intelligent retry logic, escalation
- **Invoice Generation**: Batch processing, error handling
- **Maintenance**: Data cleanup, external sync, webhook processing

## Test Configuration

### Database Setup
```python
# Isolated test database with automatic rollback
@pytest.fixture
async def db_session():
    # Creates fresh database session for each test
    # Automatic rollback ensures test isolation
```

### Mock Services
```python
# External service mocking for reliable tests
@pytest.fixture
def mock_stripe():
    # Mocks Stripe API calls
    
@pytest.fixture  
def mock_paypal():
    # Mocks PayPal API calls

@pytest.fixture
def mock_email_service():
    # Mocks email sending
```

### Test Data
```python
# Comprehensive test fixtures
@pytest.fixture
def test_customer():
    # Sample customer with full profile

@pytest.fixture
def test_subscription_active():
    # Active subscription with payment method

@pytest.fixture
def test_invoice_paid():
    # Completed invoice with payment
```

## Running Tests

### All Tests
```bash
# Run complete billing test suite
pytest tests_backend/app/billing/ -v

# Run with coverage report
pytest tests_backend/app/billing/ --cov=billing --cov-report=html
```

### Specific Test Categories
```bash
# Model tests only
pytest tests_backend/app/billing/test_models.py -v

# API endpoint tests
pytest tests_backend/app/billing/test_api.py -v

# Core engine tests
pytest tests_backend/app/billing/test_core.py -v

# Background task tests
pytest tests_backend/app/billing/test_tasks.py -v
```

### Test Patterns
```bash
# Run tests matching pattern
pytest tests_backend/app/billing/ -k "subscription" -v

# Run failed tests only
pytest tests_backend/app/billing/ --lf

# Run specific test class
pytest tests_backend/app/billing/test_models.py::TestCustomerModel -v
```

## Test Data Management

### Customer Test Data
```python
test_customers = [
    {
        "email": "test@example.com",
        "name": "Test Customer",
        "country": "FR",
        "preferred_currency": "EUR"
    }
]
```

### Plan Test Data
```python
test_plans = [
    {
        "name": "Basic Plan",
        "amount": Decimal("29.99"),
        "interval": "MONTH",
        "features": ["api_access", "basic_support"]
    }
]
```

### Payment Test Data
```python
test_payments = [
    {
        "amount": Decimal("99.99"),
        "currency": "EUR",
        "status": "SUCCEEDED",
        "provider": "STRIPE"
    }
]
```

## Performance Testing

### Load Testing
```bash
# Test with multiple concurrent users
pytest tests_backend/app/billing/test_api.py -v --numprocesses=4

# Memory profiling
pytest tests_backend/app/billing/ --profile

# Benchmark specific operations
pytest tests_backend/app/billing/test_core.py::test_payment_processing --benchmark-only
```

### Database Performance
```python
# Test query optimization
def test_subscription_query_performance():
    # Verify N+1 query prevention
    # Check index usage
    # Validate pagination performance
```

## Security Testing

### Authentication Tests
```python
def test_api_authentication():
    # Verify JWT token validation
    # Test role-based access control
    # Check API key security
```

### Data Protection Tests
```python
def test_data_encryption():
    # Verify PII encryption
    # Test payment data security
    # Check audit logging
```

## Integration Testing

### Payment Provider Integration
```python
def test_stripe_integration():
    # End-to-end payment flow
    # Webhook processing
    # Error handling
    
def test_paypal_integration():
    # Complete PayPal workflow
    # Subscription management
    # Dispute handling
```

### External Service Integration
```python
def test_email_integration():
    # Template rendering
    # Delivery tracking
    # Bounce handling

def test_pdf_generation():
    # Multi-language support
    # Template customization
    # Performance optimization
```

## Continuous Integration

### GitHub Actions
```yaml
# .github/workflows/billing-tests.yml
name: Billing System Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run billing tests
        run: pytest tests_backend/app/billing/ --cov=billing
```

### Test Coverage Requirements
- **Minimum Coverage**: 95%
- **Critical Paths**: 100% (payment processing, security)
- **Edge Cases**: Comprehensive error scenario coverage
- **Performance**: Load testing for key operations

## Best Practices

### Test Writing
1. **Descriptive Names**: Clear test method names
2. **Single Responsibility**: One assertion per test
3. **Test Independence**: No test dependencies
4. **Data Isolation**: Fresh data for each test
5. **Mock External**: Mock all external service calls

### Performance
1. **Database Optimization**: Use transactions for rollback
2. **Parallel Execution**: Independent test execution
3. **Resource Cleanup**: Proper fixture teardown
4. **Caching**: Smart use of test data caching

### Maintenance
1. **Regular Updates**: Keep tests current with code changes
2. **Refactoring**: Eliminate test code duplication
3. **Documentation**: Clear test documentation
4. **Monitoring**: Track test execution time and flakiness

## Troubleshooting

### Common Issues
```bash
# Database connection issues
export DATABASE_URL="postgresql://test:test@localhost/billing_test"

# Redis connection for caching tests
export REDIS_URL="redis://localhost:6379/1"

# Test environment variables
export STRIPE_TEST_SECRET_KEY="sk_test_..."
export PAYPAL_TEST_CLIENT_ID="test_client_id"
```

### Debug Mode
```bash
# Run tests with detailed output
pytest tests_backend/app/billing/ -v -s --tb=long

# Debug specific test
pytest tests_backend/app/billing/test_core.py::test_payment_processing -v -s --pdb
```

### Test Data Reset
```bash
# Reset test database
python -m billing.scripts.reset_test_db

# Regenerate test fixtures
python -m billing.scripts.generate_test_data
```

## Contributing

### Adding New Tests
1. Follow existing test structure and naming conventions
2. Add appropriate fixtures in `conftest.py`
3. Include both positive and negative test cases
4. Update documentation for new test categories
5. Ensure minimum 95% coverage for new code

### Test Review Checklist
- [ ] Tests are independent and isolated
- [ ] External services are properly mocked
- [ ] Error scenarios are covered
- [ ] Performance implications considered
- [ ] Documentation updated
- [ ] CI/CD pipeline passes

---

**Note**: This test suite ensures the reliability, security, and performance of the enterprise billing system. All tests should pass before deployment to production.
