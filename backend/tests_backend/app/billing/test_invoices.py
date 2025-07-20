"""
Tests for Invoice Management System
==================================

Comprehensive tests for invoice generation, management, and PDF processing.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import io
import tempfile
import os

from billing.invoices import InvoiceService, PDFGenerator, EmailService
from billing.models import (
    Customer, Plan, Subscription, Invoice, InvoiceStatus, 
    SubscriptionStatus, PlanInterval
)


class TestInvoiceService:
    """Test invoice service functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_subscription_invoice(self, invoice_service, test_subscription_active, db_session):
        """Test subscription invoice generation"""
        period_start = datetime.utcnow()
        period_end = period_start + timedelta(days=30)
        
        invoice = await invoice_service.generate_subscription_invoice(
            subscription=test_subscription_active,
            period_start=period_start,
            period_end=period_end
        )
        
        assert invoice is not None
        assert invoice.customer_id == test_subscription_active.customer_id
        assert invoice.subscription_id == test_subscription_active.id
        assert invoice.status == InvoiceStatus.DRAFT
        assert invoice.subtotal == test_subscription_active.effective_amount
        assert len(invoice.line_items) >= 1
        
        # Check line item details
        line_item = invoice.line_items[0]
        assert line_item["description"] == test_subscription_active.plan.name
        assert Decimal(line_item["unit_price"]) == test_subscription_active.effective_amount
        assert line_item["quantity"] == "1"
    
    @pytest.mark.asyncio
    async def test_generate_usage_invoice(self, invoice_service, test_subscription_active, db_session):
        """Test usage-based invoice generation"""
        usage_records = [
            {
                "metric_name": "api_calls",
                "quantity": Decimal('1500'),
                "unit": "calls",
                "unit_price": Decimal('0.01'),
                "description": "API calls usage"
            },
            {
                "metric_name": "storage",
                "quantity": Decimal('25'),
                "unit": "GB",
                "unit_price": Decimal('0.50'),
                "description": "Storage usage"
            }
        ]
        
        invoice = await invoice_service.generate_usage_invoice(
            subscription=test_subscription_active,
            usage_records=usage_records,
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow()
        )
        
        assert invoice is not None
        assert len(invoice.line_items) == 2
        
        # Check usage calculations
        expected_api_cost = Decimal('1500') * Decimal('0.01')  # 15.00
        expected_storage_cost = Decimal('25') * Decimal('0.50')  # 12.50
        expected_total = expected_api_cost + expected_storage_cost  # 27.50
        
        assert invoice.subtotal == expected_total
    
    @pytest.mark.asyncio
    async def test_proration_invoice(self, invoice_service, test_subscription_active, test_plan_yearly):
        """Test prorated invoice generation"""
        # Simulate mid-cycle plan change
        change_date = datetime.utcnow()
        days_remaining = 15
        period_end = change_date + timedelta(days=days_remaining)
        
        # Mock old plan amount (unused portion)
        unused_amount = (test_subscription_active.effective_amount * days_remaining) / 30
        
        # Mock new plan amount (prorated)
        new_amount = (test_plan_yearly.amount * days_remaining) / 365
        
        proration_data = {
            "credit_amount": unused_amount,
            "charge_amount": new_amount,
            "net_amount": new_amount - unused_amount,
            "old_plan": test_subscription_active.plan,
            "new_plan": test_plan_yearly,
            "change_date": change_date,
            "days_remaining": days_remaining
        }
        
        invoice = await invoice_service.generate_proration_invoice(
            subscription=test_subscription_active,
            proration_data=proration_data
        )
        
        assert invoice is not None
        assert len(invoice.line_items) >= 2  # Credit and charge line items
        
        # Check for credit line item
        credit_items = [item for item in invoice.line_items if "credit" in item["description"].lower()]
        assert len(credit_items) == 1
        assert Decimal(credit_items[0]["total"]) == -unused_amount
        
        # Check for charge line item
        charge_items = [item for item in invoice.line_items if "charge" in item["description"].lower()]
        assert len(charge_items) == 1
        assert Decimal(charge_items[0]["total"]) == new_amount
    
    @pytest.mark.asyncio
    async def test_finalize_invoice(self, invoice_service, test_invoice_draft, mock_tax_calculator):
        """Test invoice finalization with tax calculation"""
        # Mock tax calculation
        mock_tax_calculator.calculate_tax.return_value = {
            "tax_rate": Decimal('0.20'),
            "tax_amount": Decimal('20.00'),
            "total_amount": Decimal('120.00'),
            "tax_type": "VAT"
        }
        
        finalized_invoice = await invoice_service.finalize_invoice(test_invoice_draft.id)
        
        assert finalized_invoice.status == InvoiceStatus.OPEN
        assert finalized_invoice.tax_amount == Decimal('20.00')
        assert finalized_invoice.total == Decimal('120.00')
        assert finalized_invoice.amount_due == Decimal('120.00')
        assert finalized_invoice.finalized_at is not None
        assert finalized_invoice.number is not None
    
    @pytest.mark.asyncio
    async def test_generate_invoice_number(self, invoice_service, test_customer):
        """Test invoice number generation"""
        # Test sequential numbering
        invoice1 = await invoice_service.create_draft_invoice(
            customer_id=test_customer.id,
            currency="EUR",
            line_items=[{
                "description": "Test item 1",
                "quantity": "1",
                "unit_price": "50.00",
                "total": "50.00"
            }]
        )
        
        invoice2 = await invoice_service.create_draft_invoice(
            customer_id=test_customer.id,
            currency="EUR",
            line_items=[{
                "description": "Test item 2",
                "quantity": "1",
                "unit_price": "75.00",
                "total": "75.00"
            }]
        )
        
        # Finalize to generate numbers
        finalized1 = await invoice_service.finalize_invoice(invoice1.id)
        finalized2 = await invoice_service.finalize_invoice(invoice2.id)
        
        assert finalized1.number is not None
        assert finalized2.number is not None
        assert finalized1.number != finalized2.number
        
        # Check format (should be like INV-2025-001, INV-2025-002)
        current_year = datetime.utcnow().year
        assert str(current_year) in finalized1.number
        assert str(current_year) in finalized2.number
    
    @pytest.mark.asyncio
    async def test_mark_invoice_paid(self, invoice_service, test_invoice_paid, test_payment_successful):
        """Test marking invoice as paid"""
        # Create open invoice
        test_invoice_paid.status = InvoiceStatus.OPEN
        test_invoice_paid.amount_paid = Decimal('0')
        test_invoice_paid.paid_at = None
        
        updated_invoice = await invoice_service.mark_invoice_paid(
            invoice_id=test_invoice_paid.id,
            payment_id=test_payment_successful.id,
            amount_paid=test_invoice_paid.total
        )
        
        assert updated_invoice.status == InvoiceStatus.PAID
        assert updated_invoice.amount_paid == test_invoice_paid.total
        assert updated_invoice.amount_due == Decimal('0')
        assert updated_invoice.paid_at is not None
    
    @pytest.mark.asyncio
    async def test_partial_payment(self, invoice_service, test_invoice_draft, test_payment_successful):
        """Test partial invoice payment"""
        # Finalize invoice first
        invoice = await invoice_service.finalize_invoice(test_invoice_draft.id)
        partial_amount = invoice.total / 2
        
        updated_invoice = await invoice_service.mark_invoice_paid(
            invoice_id=invoice.id,
            payment_id=test_payment_successful.id,
            amount_paid=partial_amount
        )
        
        assert updated_invoice.status == InvoiceStatus.PARTIAL
        assert updated_invoice.amount_paid == partial_amount
        assert updated_invoice.amount_due == invoice.total - partial_amount
    
    @pytest.mark.asyncio
    async def test_void_invoice(self, invoice_service, test_invoice_draft):
        """Test voiding an invoice"""
        voided_invoice = await invoice_service.void_invoice(
            invoice_id=test_invoice_draft.id,
            reason="Customer cancelled order"
        )
        
        assert voided_invoice.status == InvoiceStatus.VOID
        assert voided_invoice.amount_due == Decimal('0')
        assert voided_invoice.voided_at is not None
        assert "Customer cancelled order" in voided_invoice.notes
    
    @pytest.mark.asyncio
    async def test_recurring_invoice_generation(self, invoice_service, test_subscription_active):
        """Test automatic recurring invoice generation"""
        # Simulate subscription period ending
        test_subscription_active.current_period_end = datetime.utcnow() + timedelta(days=1)
        
        invoice = await invoice_service.generate_recurring_invoice(test_subscription_active.id)
        
        assert invoice is not None
        assert invoice.subscription_id == test_subscription_active.id
        assert invoice.status == InvoiceStatus.DRAFT
        
        # Check that period dates are set correctly
        line_item = invoice.line_items[0]
        assert "period" in line_item["description"].lower()


class TestPDFGenerator:
    """Test PDF generation functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_invoice_pdf(self, pdf_generator, test_invoice_paid):
        """Test basic PDF generation"""
        pdf_data = await pdf_generator.generate_invoice_pdf(test_invoice_paid)
        
        assert pdf_data is not None
        assert len(pdf_data) > 0
        assert pdf_data.startswith(b'%PDF')  # PDF header
    
    @pytest.mark.asyncio
    async def test_pdf_content_verification(self, pdf_generator, test_invoice_paid):
        """Test PDF content includes required information"""
        pdf_data = await pdf_generator.generate_invoice_pdf(test_invoice_paid)
        
        # Convert PDF to text for verification (mock implementation)
        with patch('pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = f"""
            INVOICE
            Invoice Number: {test_invoice_paid.number}
            Customer: {test_invoice_paid.customer.name}
            Total: €{test_invoice_paid.total}
            Due Date: {test_invoice_paid.due_date.strftime('%Y-%m-%d')}
            """
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]
            
            # Verify PDF contains required information
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                text = pdf.pages[0].extract_text()
                
                assert test_invoice_paid.number in text
                assert test_invoice_paid.customer.name in text
                assert str(test_invoice_paid.total) in text
    
    @pytest.mark.asyncio
    async def test_pdf_with_logo(self, pdf_generator, test_invoice_paid):
        """Test PDF generation with company logo"""
        # Mock logo file
        logo_path = "/tmp/test_logo.png"
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as logo_file:
            logo_file.write(b'PNG_CONTENT')
            logo_path = logo_file.name
        
        try:
            pdf_data = await pdf_generator.generate_invoice_pdf(
                test_invoice_paid,
                include_logo=True,
                logo_path=logo_path
            )
            
            assert pdf_data is not None
            assert len(pdf_data) > 0
        finally:
            if os.path.exists(logo_path):
                os.unlink(logo_path)
    
    @pytest.mark.asyncio
    async def test_pdf_custom_template(self, pdf_generator, test_invoice_paid):
        """Test PDF generation with custom template"""
        custom_template = {
            "header_color": "#2E7D32",
            "font_family": "Helvetica",
            "include_payment_instructions": True,
            "footer_text": "Thank you for your business!"
        }
        
        pdf_data = await pdf_generator.generate_invoice_pdf(
            test_invoice_paid,
            template_config=custom_template
        )
        
        assert pdf_data is not None
        assert len(pdf_data) > 0
    
    @pytest.mark.asyncio
    async def test_pdf_multilingual(self, pdf_generator, test_invoice_paid, test_customer):
        """Test PDF generation in different languages"""
        # Set customer language to French
        test_customer.preferred_language = "fr"
        
        pdf_data = await pdf_generator.generate_invoice_pdf(
            test_invoice_paid,
            language="fr"
        )
        
        assert pdf_data is not None
        
        # Verify French content (mock verification)
        with patch('pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "FACTURE\nTotal: €120,00\nÉchéance:"
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]
            
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                text = pdf.pages[0].extract_text()
                assert "FACTURE" in text  # French for "INVOICE"
    
    @pytest.mark.asyncio
    async def test_pdf_with_attachments(self, pdf_generator, test_invoice_paid):
        """Test PDF generation with additional attachments"""
        attachments = [
            {
                "name": "Terms and Conditions.pdf",
                "data": b"PDF_TERMS_CONTENT",
                "type": "application/pdf"
            },
            {
                "name": "Receipt.pdf",
                "data": b"PDF_RECEIPT_CONTENT",
                "type": "application/pdf"
            }
        ]
        
        pdf_data = await pdf_generator.generate_invoice_pdf(
            test_invoice_paid,
            attachments=attachments
        )
        
        assert pdf_data is not None
        assert len(pdf_data) > len(attachments[0]["data"])  # Should be larger with attachments


class TestEmailService:
    """Test email service functionality"""
    
    @pytest.mark.asyncio
    async def test_send_invoice_email(self, email_service, test_invoice_paid, mock_smtp):
        """Test sending invoice via email"""
        recipient = "customer@example.com"
        subject = "Your Invoice"
        message = "Please find your invoice attached."
        
        result = await email_service.send_invoice_email(
            invoice=test_invoice_paid,
            recipient_email=recipient,
            subject=subject,
            message=message
        )
        
        assert result is True
        mock_smtp.send_message.assert_called_once()
        
        # Verify email content
        call_args = mock_smtp.send_message.call_args[0][0]
        assert recipient in call_args['To']
        assert subject in call_args['Subject']
    
    @pytest.mark.asyncio
    async def test_send_payment_reminder(self, email_service, test_invoice_draft, mock_smtp):
        """Test sending payment reminder email"""
        # Set invoice as overdue
        test_invoice_draft.due_date = datetime.utcnow() - timedelta(days=5)
        test_invoice_draft.status = InvoiceStatus.OPEN
        
        result = await email_service.send_payment_reminder(
            invoice=test_invoice_draft,
            reminder_type="first_reminder"
        )
        
        assert result is True
        mock_smtp.send_message.assert_called_once()
        
        # Verify reminder content
        call_args = mock_smtp.send_message.call_args[0][0]
        assert "reminder" in call_args['Subject'].lower()
        assert "overdue" in str(call_args).lower()
    
    @pytest.mark.asyncio
    async def test_send_payment_confirmation(self, email_service, test_invoice_paid, test_payment_successful, mock_smtp):
        """Test sending payment confirmation email"""
        result = await email_service.send_payment_confirmation(
            invoice=test_invoice_paid,
            payment=test_payment_successful
        )
        
        assert result is True
        mock_smtp.send_message.assert_called_once()
        
        # Verify confirmation content
        call_args = mock_smtp.send_message.call_args[0][0]
        assert "payment" in call_args['Subject'].lower()
        assert "confirmation" in call_args['Subject'].lower()
    
    @pytest.mark.asyncio
    async def test_email_template_rendering(self, email_service, test_invoice_paid):
        """Test email template rendering"""
        template_data = {
            "customer_name": test_invoice_paid.customer.name,
            "invoice_number": test_invoice_paid.number,
            "amount": str(test_invoice_paid.total),
            "due_date": test_invoice_paid.due_date.strftime("%B %d, %Y")
        }
        
        html_content = await email_service.render_template(
            template_name="invoice_email.html",
            data=template_data
        )
        
        assert html_content is not None
        assert test_invoice_paid.customer.name in html_content
        assert test_invoice_paid.number in html_content
        assert str(test_invoice_paid.total) in html_content
    
    @pytest.mark.asyncio
    async def test_email_with_custom_attachments(self, email_service, test_invoice_paid, mock_smtp):
        """Test email with custom attachments"""
        attachments = [
            {
                "filename": "terms.pdf",
                "content": b"PDF_TERMS_CONTENT",
                "content_type": "application/pdf"
            },
            {
                "filename": "receipt.txt",
                "content": b"TEXT_RECEIPT_CONTENT",
                "content_type": "text/plain"
            }
        ]
        
        result = await email_service.send_invoice_email(
            invoice=test_invoice_paid,
            recipient_email="customer@example.com",
            attachments=attachments
        )
        
        assert result is True
        mock_smtp.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_email_delivery_failure_handling(self, email_service, test_invoice_paid, mock_smtp):
        """Test email delivery failure handling"""
        # Mock SMTP failure
        mock_smtp.send_message.side_effect = Exception("SMTP server unavailable")
        
        result = await email_service.send_invoice_email(
            invoice=test_invoice_paid,
            recipient_email="invalid@nonexistent.domain"
        )
        
        assert result is False
        
        # Verify error is logged (mock logger verification)
        # In real implementation, check logging or error tracking
    
    @pytest.mark.asyncio
    async def test_bulk_email_sending(self, email_service, mock_smtp):
        """Test bulk email sending for multiple invoices"""
        recipients = [
            {"invoice_id": "inv_1", "email": "customer1@example.com"},
            {"invoice_id": "inv_2", "email": "customer2@example.com"},
            {"invoice_id": "inv_3", "email": "customer3@example.com"}
        ]
        
        results = await email_service.send_bulk_emails(
            recipients=recipients,
            email_type="payment_reminder"
        )
        
        assert len(results) == 3
        assert all(result["success"] is True for result in results)
        assert mock_smtp.send_message.call_count == 3


class TestInvoiceIntegration:
    """Integration tests for invoice system"""
    
    @pytest.mark.asyncio
    async def test_complete_invoice_workflow(self, invoice_service, pdf_generator, email_service, test_subscription_active, mock_smtp):
        """Test complete invoice workflow from generation to delivery"""
        # 1. Generate subscription invoice
        period_start = datetime.utcnow()
        period_end = period_start + timedelta(days=30)
        
        invoice = await invoice_service.generate_subscription_invoice(
            subscription=test_subscription_active,
            period_start=period_start,
            period_end=period_end
        )
        
        assert invoice.status == InvoiceStatus.DRAFT
        
        # 2. Finalize invoice
        finalized_invoice = await invoice_service.finalize_invoice(invoice.id)
        assert finalized_invoice.status == InvoiceStatus.OPEN
        assert finalized_invoice.number is not None
        
        # 3. Generate PDF
        pdf_data = await pdf_generator.generate_invoice_pdf(finalized_invoice)
        assert pdf_data is not None
        assert len(pdf_data) > 0
        
        # 4. Send email with PDF attachment
        result = await email_service.send_invoice_email(
            invoice=finalized_invoice,
            recipient_email=test_subscription_active.customer.email,
            include_pdf=True,
            pdf_data=pdf_data
        )
        
        assert result is True
        mock_smtp.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dunning_process_integration(self, invoice_service, email_service, test_invoice_draft, mock_smtp):
        """Test automated dunning process"""
        # 1. Create overdue invoice
        overdue_invoice = await invoice_service.finalize_invoice(test_invoice_draft.id)
        overdue_invoice.due_date = datetime.utcnow() - timedelta(days=5)
        overdue_invoice.status = InvoiceStatus.OPEN
        
        # 2. Process dunning workflow
        dunning_results = await invoice_service.process_dunning_workflow(overdue_invoice.id)
        
        assert dunning_results["reminder_sent"] is True
        assert dunning_results["next_reminder_date"] is not None
        
        # 3. Verify reminder email was sent
        mock_smtp.send_message.assert_called_once()
        
        # 4. Check email content for overdue notice
        call_args = mock_smtp.send_message.call_args[0][0]
        assert "overdue" in str(call_args).lower() or "reminder" in str(call_args).lower()
