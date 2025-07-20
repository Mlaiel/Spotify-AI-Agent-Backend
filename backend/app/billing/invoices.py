"""
Spotify AI Agent - Invoice Management System
===========================================

Professional invoice generation, management, and delivery system.
Supports multi-language, multi-currency, and various tax jurisdictions.

Features:
- PDF invoice generation with custom templates
- Multi-language support (EN/FR/DE/ES)
- Tax compliance (VAT, Sales Tax, GST)
- Email delivery with tracking
- Invoice archival and retrieval
- Dunning management for overdue invoices
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, List, Union, BinaryIO
from enum import Enum
from dataclasses import dataclass, field
import uuid
import json
import tempfile
from pathlib import Path
import aiofiles
import aioredis
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML, CSS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formataddr
import boto3
from botocore.exceptions import ClientError


class InvoiceStatus(Enum):
    """Invoice status enumeration"""
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class InvoiceType(Enum):
    """Invoice type enumeration"""
    SUBSCRIPTION = "subscription"
    ONE_TIME = "one_time"
    USAGE_BASED = "usage_based"
    CREDIT_NOTE = "credit_note"
    PROFORMA = "proforma"


class DeliveryMethod(Enum):
    """Invoice delivery methods"""
    EMAIL = "email"
    POSTAL = "postal"
    PORTAL = "portal"
    API = "api"


@dataclass
class InvoiceLineItem:
    """Individual line item on an invoice"""
    description: str
    quantity: Decimal
    unit_price: Decimal
    tax_rate: Decimal
    discount_rate: Decimal = Decimal('0')
    product_code: Optional[str] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    @property
    def subtotal(self) -> Decimal:
        """Calculate line item subtotal before tax"""
        gross = self.quantity * self.unit_price
        discount = gross * (self.discount_rate / 100)
        return gross - discount
    
    @property
    def tax_amount(self) -> Decimal:
        """Calculate tax amount for line item"""
        return self.subtotal * (self.tax_rate / 100)
    
    @property
    def total(self) -> Decimal:
        """Calculate total including tax"""
        return self.subtotal + self.tax_amount


@dataclass
class InvoiceRecipient:
    """Invoice recipient information"""
    name: str
    email: str
    company: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    tax_id: Optional[str] = None
    language: str = "en"


@dataclass
class InvoiceData:
    """Complete invoice data structure"""
    id: str
    number: str
    status: InvoiceStatus
    type: InvoiceType
    recipient: InvoiceRecipient
    line_items: List[InvoiceLineItem]
    issue_date: datetime
    due_date: datetime
    currency: str
    notes: Optional[str] = None
    payment_terms: Optional[str] = None
    po_number: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def subtotal(self) -> Decimal:
        """Calculate invoice subtotal"""
        return sum(item.subtotal for item in self.line_items)
    
    @property
    def tax_total(self) -> Decimal:
        """Calculate total tax amount"""
        return sum(item.tax_amount for item in self.line_items)
    
    @property
    def total(self) -> Decimal:
        """Calculate invoice total"""
        return self.subtotal + self.tax_total


class InvoiceNumberGenerator:
    """Generates sequential invoice numbers"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.prefix = "INV"
        
    async def generate_number(self, year: Optional[int] = None) -> str:
        """Generate next invoice number"""
        if year is None:
            year = datetime.utcnow().year
        
        counter_key = f"invoice_counter:{year}"
        counter = await self.redis_client.incr(counter_key)
        
        # Set expiration at end of year
        if counter == 1:
            year_end = datetime(year + 1, 1, 1)
            await self.redis_client.expireat(counter_key, int(year_end.timestamp()))
        
        return f"{self.prefix}-{year}-{counter:06d}"


class InvoiceTemplateEngine:
    """Template engine for invoice generation"""
    
    def __init__(self, template_dir: str = None):
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(__file__), "templates"
        )
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True
        )
        self.logger = logging.getLogger(__name__)
        
    def render_html(self, invoice_data: InvoiceData, template_name: str = None) -> str:
        """Render invoice as HTML"""
        try:
            # Select template based on language and type
            if template_name is None:
                template_name = self._select_template(invoice_data)
            
            template = self.env.get_template(template_name)
            
            # Prepare template context
            context = {
                'invoice': invoice_data,
                'company': self._get_company_info(),
                'formatting': self._get_formatting_helpers(invoice_data.currency),
                'translations': self._get_translations(invoice_data.recipient.language)
            }
            
            return template.render(**context)
            
        except Exception as exc:
            self.logger.error(f"Template rendering failed: {exc}")
            raise
    
    def _select_template(self, invoice_data: InvoiceData) -> str:
        """Select appropriate template"""
        lang = invoice_data.recipient.language
        invoice_type = invoice_data.type.value
        
        # Try specific template first
        specific_template = f"invoice_{invoice_type}_{lang}.html"
        if os.path.exists(os.path.join(self.template_dir, specific_template)):
            return specific_template
        
        # Fall back to generic template
        generic_template = f"invoice_{lang}.html"
        if os.path.exists(os.path.join(self.template_dir, generic_template)):
            return generic_template
        
        # Final fallback
        return "invoice_en.html"
    
    def _get_company_info(self) -> Dict[str, str]:
        """Get company information for invoice"""
        return {
            'name': 'Spotify AI Agent',
            'address_line1': '123 Music Street',
            'address_line2': 'Suite 456',
            'city': 'Paris',
            'postal_code': '75001',
            'country': 'France',
            'phone': '+33 1 23 45 67 89',
            'email': 'billing@spotify-ai.com',
            'website': 'https://spotify-ai.com',
            'tax_id': 'FR12345678901',
            'siret': '12345678901234'
        }
    
    def _get_formatting_helpers(self, currency: str) -> Dict[str, Any]:
        """Get formatting helpers for templates"""
        return {
            'currency': currency,
            'currency_symbol': self._get_currency_symbol(currency),
            'date_format': '%d/%m/%Y',
            'decimal_places': 2
        }
    
    def _get_currency_symbol(self, currency: str) -> str:
        """Get currency symbol"""
        symbols = {
            'EUR': '€',
            'USD': '$',
            'GBP': '£',
            'JPY': '¥'
        }
        return symbols.get(currency, currency)
    
    def _get_translations(self, language: str) -> Dict[str, str]:
        """Get translations for template"""
        translations = {
            'en': {
                'invoice': 'Invoice',
                'invoice_number': 'Invoice Number',
                'issue_date': 'Issue Date',
                'due_date': 'Due Date',
                'bill_to': 'Bill To',
                'description': 'Description',
                'quantity': 'Qty',
                'unit_price': 'Unit Price',
                'total': 'Total',
                'subtotal': 'Subtotal',
                'tax': 'Tax',
                'amount_due': 'Amount Due',
                'payment_terms': 'Payment Terms',
                'notes': 'Notes'
            },
            'fr': {
                'invoice': 'Facture',
                'invoice_number': 'Numéro de facture',
                'issue_date': 'Date d\'émission',
                'due_date': 'Date d\'échéance',
                'bill_to': 'Facturer à',
                'description': 'Description',
                'quantity': 'Qté',
                'unit_price': 'Prix unitaire',
                'total': 'Total',
                'subtotal': 'Sous-total',
                'tax': 'TVA',
                'amount_due': 'Montant dû',
                'payment_terms': 'Conditions de paiement',
                'notes': 'Notes'
            },
            'de': {
                'invoice': 'Rechnung',
                'invoice_number': 'Rechnungsnummer',
                'issue_date': 'Ausstellungsdatum',
                'due_date': 'Fälligkeitsdatum',
                'bill_to': 'Rechnung an',
                'description': 'Beschreibung',
                'quantity': 'Menge',
                'unit_price': 'Einzelpreis',
                'total': 'Gesamt',
                'subtotal': 'Zwischensumme',
                'tax': 'MwSt',
                'amount_due': 'Fälliger Betrag',
                'payment_terms': 'Zahlungsbedingungen',
                'notes': 'Notizen'
            }
        }
        
        return translations.get(language, translations['en'])


class InvoicePDFGenerator:
    """PDF generation from HTML invoices"""
    
    def __init__(self, template_engine: InvoiceTemplateEngine):
        self.template_engine = template_engine
        self.logger = logging.getLogger(__name__)
        
    async def generate_pdf(self, invoice_data: InvoiceData) -> bytes:
        """Generate PDF from invoice data"""
        try:
            # Render HTML
            html_content = self.template_engine.render_html(invoice_data)
            
            # Generate PDF
            pdf_bytes = await self._html_to_pdf(html_content)
            
            return pdf_bytes
            
        except Exception as exc:
            self.logger.error(f"PDF generation failed: {exc}")
            raise
    
    async def _html_to_pdf(self, html_content: str) -> bytes:
        """Convert HTML to PDF using WeasyPrint"""
        try:
            # CSS for PDF styling
            css_content = """
            @page {
                size: A4;
                margin: 2cm;
            }
            body {
                font-family: 'Helvetica', 'Arial', sans-serif;
                font-size: 10pt;
                line-height: 1.4;
            }
            .header {
                margin-bottom: 2cm;
            }
            .invoice-table {
                width: 100%;
                border-collapse: collapse;
            }
            .invoice-table th,
            .invoice-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .total-row {
                font-weight: bold;
                background-color: #f9f9f9;
            }
            """
            
            # Create HTML and CSS objects
            html_obj = HTML(string=html_content)
            css_obj = CSS(string=css_content)
            
            # Generate PDF
            pdf_bytes = html_obj.write_pdf(stylesheets=[css_obj])
            
            return pdf_bytes
            
        except Exception as exc:
            self.logger.error(f"HTML to PDF conversion failed: {exc}")
            raise


class InvoiceStorageManager:
    """Manages invoice storage and retrieval"""
    
    def __init__(self, storage_backend: str = "s3"):
        self.storage_backend = storage_backend
        self.logger = logging.getLogger(__name__)
        
        if storage_backend == "s3":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'eu-west-1')
            )
            self.bucket_name = os.getenv('INVOICE_BUCKET', 'spotify-ai-invoices')
    
    async def store_invoice_pdf(self, invoice_id: str, pdf_data: bytes) -> str:
        """Store invoice PDF and return storage key"""
        try:
            if self.storage_backend == "s3":
                return await self._store_s3(invoice_id, pdf_data)
            else:
                return await self._store_local(invoice_id, pdf_data)
                
        except Exception as exc:
            self.logger.error(f"Invoice storage failed: {exc}")
            raise
    
    async def retrieve_invoice_pdf(self, storage_key: str) -> bytes:
        """Retrieve invoice PDF from storage"""
        try:
            if self.storage_backend == "s3":
                return await self._retrieve_s3(storage_key)
            else:
                return await self._retrieve_local(storage_key)
                
        except Exception as exc:
            self.logger.error(f"Invoice retrieval failed: {exc}")
            raise
    
    async def _store_s3(self, invoice_id: str, pdf_data: bytes) -> str:
        """Store PDF in S3"""
        try:
            # Generate storage key with date partitioning
            now = datetime.utcnow()
            storage_key = f"invoices/{now.year}/{now.month:02d}/{invoice_id}.pdf"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=storage_key,
                Body=pdf_data,
                ContentType='application/pdf',
                Metadata={
                    'invoice_id': invoice_id,
                    'created_at': now.isoformat()
                }
            )
            
            return storage_key
            
        except ClientError as exc:
            self.logger.error(f"S3 upload failed: {exc}")
            raise
    
    async def _retrieve_s3(self, storage_key: str) -> bytes:
        """Retrieve PDF from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=storage_key
            )
            
            return response['Body'].read()
            
        except ClientError as exc:
            self.logger.error(f"S3 download failed: {exc}")
            raise
    
    async def _store_local(self, invoice_id: str, pdf_data: bytes) -> str:
        """Store PDF locally"""
        try:
            # Create directory structure
            base_dir = Path(os.getenv('INVOICE_STORAGE_DIR', './invoices'))
            now = datetime.utcnow()
            invoice_dir = base_dir / str(now.year) / f"{now.month:02d}"
            invoice_dir.mkdir(parents=True, exist_ok=True)
            
            # Write file
            file_path = invoice_dir / f"{invoice_id}.pdf"
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(pdf_data)
            
            return str(file_path.relative_to(base_dir))
            
        except Exception as exc:
            self.logger.error(f"Local storage failed: {exc}")
            raise
    
    async def _retrieve_local(self, storage_key: str) -> bytes:
        """Retrieve PDF from local storage"""
        try:
            base_dir = Path(os.getenv('INVOICE_STORAGE_DIR', './invoices'))
            file_path = base_dir / storage_key
            
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
                
        except Exception as exc:
            self.logger.error(f"Local retrieval failed: {exc}")
            raise


class InvoiceEmailDelivery:
    """Email delivery system for invoices"""
    
    def __init__(self):
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.from_email = os.getenv('FROM_EMAIL', 'billing@spotify-ai.com')
        self.from_name = os.getenv('FROM_NAME', 'Spotify AI Agent')
        self.logger = logging.getLogger(__name__)
    
    async def send_invoice(self, invoice_data: InvoiceData, pdf_data: bytes) -> bool:
        """Send invoice via email"""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = formataddr((self.from_name, self.from_email))
            msg['To'] = invoice_data.recipient.email
            msg['Subject'] = self._get_subject(invoice_data)
            
            # Add body
            body_text = self._get_email_body(invoice_data)
            msg.attach(MIMEText(body_text, 'html'))
            
            # Add PDF attachment
            pdf_attachment = MIMEApplication(pdf_data, _subtype='pdf')
            pdf_attachment.add_header(
                'Content-Disposition', 
                'attachment', 
                filename=f"invoice_{invoice_data.number}.pdf"
            )
            msg.attach(pdf_attachment)
            
            # Send email
            await self._send_email(msg)
            
            return True
            
        except Exception as exc:
            self.logger.error(f"Email delivery failed: {exc}")
            return False
    
    def _get_subject(self, invoice_data: InvoiceData) -> str:
        """Generate email subject"""
        lang = invoice_data.recipient.language
        
        subjects = {
            'en': f"Invoice {invoice_data.number} from Spotify AI Agent",
            'fr': f"Facture {invoice_data.number} de Spotify AI Agent",
            'de': f"Rechnung {invoice_data.number} von Spotify AI Agent"
        }
        
        return subjects.get(lang, subjects['en'])
    
    def _get_email_body(self, invoice_data: InvoiceData) -> str:
        """Generate email body"""
        lang = invoice_data.recipient.language
        
        bodies = {
            'en': f"""
            <html>
            <body>
                <p>Dear {invoice_data.recipient.name},</p>
                
                <p>Please find attached your invoice <strong>{invoice_data.number}</strong> 
                for Spotify AI Agent services.</p>
                
                <p><strong>Invoice Details:</strong></p>
                <ul>
                    <li>Invoice Number: {invoice_data.number}</li>
                    <li>Issue Date: {invoice_data.issue_date.strftime('%B %d, %Y')}</li>
                    <li>Due Date: {invoice_data.due_date.strftime('%B %d, %Y')}</li>
                    <li>Amount Due: {invoice_data.total} {invoice_data.currency}</li>
                </ul>
                
                <p>Payment can be made through your account dashboard or by following 
                the payment instructions in the attached invoice.</p>
                
                <p>If you have any questions, please don't hesitate to contact our 
                billing support team.</p>
                
                <p>Best regards,<br>
                The Spotify AI Agent Team</p>
            </body>
            </html>
            """,
            'fr': f"""
            <html>
            <body>
                <p>Cher/Chère {invoice_data.recipient.name},</p>
                
                <p>Veuillez trouver ci-joint votre facture <strong>{invoice_data.number}</strong> 
                pour les services Spotify AI Agent.</p>
                
                <p><strong>Détails de la facture :</strong></p>
                <ul>
                    <li>Numéro de facture : {invoice_data.number}</li>
                    <li>Date d'émission : {invoice_data.issue_date.strftime('%d %B %Y')}</li>
                    <li>Date d'échéance : {invoice_data.due_date.strftime('%d %B %Y')}</li>
                    <li>Montant dû : {invoice_data.total} {invoice_data.currency}</li>
                </ul>
                
                <p>Le paiement peut être effectué via votre tableau de bord ou en suivant 
                les instructions de paiement dans la facture ci-jointe.</p>
                
                <p>Si vous avez des questions, n'hésitez pas à contacter notre équipe 
                de support facturation.</p>
                
                <p>Cordialement,<br>
                L'équipe Spotify AI Agent</p>
            </body>
            </html>
            """
        }
        
        return bodies.get(lang, bodies['en'])
    
    async def _send_email(self, msg: MIMEMultipart):
        """Send email via SMTP"""
        try:
            # Use asyncio to run SMTP in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._smtp_send, msg)
            
        except Exception as exc:
            self.logger.error(f"SMTP send failed: {exc}")
            raise
    
    def _smtp_send(self, msg: MIMEMultipart):
        """Send email synchronously"""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)


class InvoiceManager:
    """Main invoice management orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        
        # Initialize components
        self.number_generator = None
        self.template_engine = InvoiceTemplateEngine()
        self.pdf_generator = InvoicePDFGenerator(self.template_engine)
        self.storage_manager = InvoiceStorageManager()
        self.email_delivery = InvoiceEmailDelivery()
    
    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await aioredis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379')
        )
        self.number_generator = InvoiceNumberGenerator(self.redis_client)
    
    async def create_invoice(self, 
                           recipient: InvoiceRecipient,
                           line_items: List[InvoiceLineItem],
                           invoice_type: InvoiceType = InvoiceType.ONE_TIME,
                           currency: str = "EUR",
                           payment_terms: str = "Net 30",
                           notes: str = None,
                           metadata: Dict[str, Any] = None) -> InvoiceData:
        """Create a new invoice"""
        try:
            # Generate invoice ID and number
            invoice_id = str(uuid.uuid4())
            invoice_number = await self.number_generator.generate_number()
            
            # Calculate dates
            issue_date = datetime.utcnow()
            due_date = issue_date + timedelta(days=30)  # Default to 30 days
            
            # Create invoice data
            invoice_data = InvoiceData(
                id=invoice_id,
                number=invoice_number,
                status=InvoiceStatus.DRAFT,
                type=invoice_type,
                recipient=recipient,
                line_items=line_items,
                issue_date=issue_date,
                due_date=due_date,
                currency=currency,
                payment_terms=payment_terms,
                notes=notes,
                metadata=metadata or {}
            )
            
            # Store invoice data
            await self._store_invoice_data(invoice_data)
            
            self.logger.info(f"Invoice created: {invoice_id} ({invoice_number})")
            
            return invoice_data
            
        except Exception as exc:
            self.logger.error(f"Invoice creation failed: {exc}")
            raise
    
    async def generate_pdf(self, invoice_id: str) -> bytes:
        """Generate PDF for an invoice"""
        try:
            # Get invoice data
            invoice_data = await self._get_invoice_data(invoice_id)
            if not invoice_data:
                raise ValueError(f"Invoice not found: {invoice_id}")
            
            # Generate PDF
            pdf_data = await self.pdf_generator.generate_pdf(invoice_data)
            
            # Store PDF
            storage_key = await self.storage_manager.store_invoice_pdf(invoice_id, pdf_data)
            
            # Update invoice with storage info
            invoice_data.metadata['pdf_storage_key'] = storage_key
            await self._store_invoice_data(invoice_data)
            
            return pdf_data
            
        except Exception as exc:
            self.logger.error(f"PDF generation failed for invoice {invoice_id}: {exc}")
            raise
    
    async def send_invoice(self, invoice_id: str, 
                          delivery_method: DeliveryMethod = DeliveryMethod.EMAIL) -> bool:
        """Send invoice to recipient"""
        try:
            # Get invoice data
            invoice_data = await self._get_invoice_data(invoice_id)
            if not invoice_data:
                raise ValueError(f"Invoice not found: {invoice_id}")
            
            # Generate PDF if not exists
            if 'pdf_storage_key' not in invoice_data.metadata:
                await self.generate_pdf(invoice_id)
                invoice_data = await self._get_invoice_data(invoice_id)  # Refresh
            
            # Get PDF data
            pdf_data = await self.storage_manager.retrieve_invoice_pdf(
                invoice_data.metadata['pdf_storage_key']
            )
            
            # Send via requested method
            success = False
            if delivery_method == DeliveryMethod.EMAIL:
                success = await self.email_delivery.send_invoice(invoice_data, pdf_data)
            
            if success:
                # Update status
                invoice_data.status = InvoiceStatus.SENT
                invoice_data.metadata['sent_at'] = datetime.utcnow().isoformat()
                invoice_data.metadata['delivery_method'] = delivery_method.value
                await self._store_invoice_data(invoice_data)
                
                self.logger.info(f"Invoice sent: {invoice_id}")
            
            return success
            
        except Exception as exc:
            self.logger.error(f"Invoice sending failed for {invoice_id}: {exc}")
            return False
    
    async def mark_paid(self, invoice_id: str, payment_id: str, 
                       payment_date: datetime = None) -> bool:
        """Mark invoice as paid"""
        try:
            invoice_data = await self._get_invoice_data(invoice_id)
            if not invoice_data:
                raise ValueError(f"Invoice not found: {invoice_id}")
            
            # Update status
            invoice_data.status = InvoiceStatus.PAID
            invoice_data.metadata['paid_at'] = (payment_date or datetime.utcnow()).isoformat()
            invoice_data.metadata['payment_id'] = payment_id
            
            await self._store_invoice_data(invoice_data)
            
            self.logger.info(f"Invoice marked as paid: {invoice_id}")
            return True
            
        except Exception as exc:
            self.logger.error(f"Failed to mark invoice as paid {invoice_id}: {exc}")
            return False
    
    async def get_invoice(self, invoice_id: str) -> Optional[InvoiceData]:
        """Retrieve invoice data"""
        return await self._get_invoice_data(invoice_id)
    
    async def list_invoices(self, customer_id: str = None, 
                           status: InvoiceStatus = None,
                           limit: int = 50) -> List[InvoiceData]:
        """List invoices with optional filtering"""
        try:
            # Get invoice IDs from Redis
            if customer_id:
                invoice_ids = await self.redis_client.lrange(
                    f"customer_invoices:{customer_id}", 0, limit - 1
                )
            else:
                invoice_ids = await self.redis_client.lrange("all_invoices", 0, limit - 1)
            
            # Get invoice data
            invoices = []
            for invoice_id in invoice_ids:
                if isinstance(invoice_id, bytes):
                    invoice_id = invoice_id.decode('utf-8')
                
                invoice_data = await self._get_invoice_data(invoice_id)
                if invoice_data and (not status or invoice_data.status == status):
                    invoices.append(invoice_data)
            
            return invoices
            
        except Exception as exc:
            self.logger.error(f"Invoice listing failed: {exc}")
            return []
    
    async def _store_invoice_data(self, invoice_data: InvoiceData):
        """Store invoice data in Redis"""
        try:
            # Convert to dict for JSON serialization
            data_dict = {
                'id': invoice_data.id,
                'number': invoice_data.number,
                'status': invoice_data.status.value,
                'type': invoice_data.type.value,
                'recipient': {
                    'name': invoice_data.recipient.name,
                    'email': invoice_data.recipient.email,
                    'company': invoice_data.recipient.company,
                    'address_line1': invoice_data.recipient.address_line1,
                    'address_line2': invoice_data.recipient.address_line2,
                    'city': invoice_data.recipient.city,
                    'state': invoice_data.recipient.state,
                    'postal_code': invoice_data.recipient.postal_code,
                    'country': invoice_data.recipient.country,
                    'tax_id': invoice_data.recipient.tax_id,
                    'language': invoice_data.recipient.language
                },
                'line_items': [
                    {
                        'description': item.description,
                        'quantity': str(item.quantity),
                        'unit_price': str(item.unit_price),
                        'tax_rate': str(item.tax_rate),
                        'discount_rate': str(item.discount_rate),
                        'product_code': item.product_code,
                        'period_start': item.period_start.isoformat() if item.period_start else None,
                        'period_end': item.period_end.isoformat() if item.period_end else None
                    }
                    for item in invoice_data.line_items
                ],
                'issue_date': invoice_data.issue_date.isoformat(),
                'due_date': invoice_data.due_date.isoformat(),
                'currency': invoice_data.currency,
                'notes': invoice_data.notes,
                'payment_terms': invoice_data.payment_terms,
                'po_number': invoice_data.po_number,
                'created_at': invoice_data.created_at.isoformat(),
                'updated_at': invoice_data.updated_at.isoformat(),
                'metadata': invoice_data.metadata
            }
            
            # Store invoice
            await self.redis_client.setex(
                f"invoice:{invoice_data.id}",
                86400 * 365,  # 1 year
                json.dumps(data_dict, default=str)
            )
            
            # Update indexes
            await self.redis_client.lpush("all_invoices", invoice_data.id)
            
            # Update customer index if we can determine customer ID
            customer_id = invoice_data.metadata.get('customer_id')
            if customer_id:
                await self.redis_client.lpush(f"customer_invoices:{customer_id}", invoice_data.id)
            
        except Exception as exc:
            self.logger.error(f"Invoice storage failed: {exc}")
            raise
    
    async def _get_invoice_data(self, invoice_id: str) -> Optional[InvoiceData]:
        """Retrieve invoice data from Redis"""
        try:
            data = await self.redis_client.get(f"invoice:{invoice_id}")
            if not data:
                return None
            
            data_dict = json.loads(data)
            
            # Reconstruct objects
            recipient = InvoiceRecipient(**data_dict['recipient'])
            
            line_items = []
            for item_data in data_dict['line_items']:
                line_item = InvoiceLineItem(
                    description=item_data['description'],
                    quantity=Decimal(item_data['quantity']),
                    unit_price=Decimal(item_data['unit_price']),
                    tax_rate=Decimal(item_data['tax_rate']),
                    discount_rate=Decimal(item_data['discount_rate']),
                    product_code=item_data['product_code'],
                    period_start=datetime.fromisoformat(item_data['period_start']) if item_data['period_start'] else None,
                    period_end=datetime.fromisoformat(item_data['period_end']) if item_data['period_end'] else None
                )
                line_items.append(line_item)
            
            invoice_data = InvoiceData(
                id=data_dict['id'],
                number=data_dict['number'],
                status=InvoiceStatus(data_dict['status']),
                type=InvoiceType(data_dict['type']),
                recipient=recipient,
                line_items=line_items,
                issue_date=datetime.fromisoformat(data_dict['issue_date']),
                due_date=datetime.fromisoformat(data_dict['due_date']),
                currency=data_dict['currency'],
                notes=data_dict['notes'],
                payment_terms=data_dict['payment_terms'],
                po_number=data_dict['po_number'],
                created_at=datetime.fromisoformat(data_dict['created_at']),
                updated_at=datetime.fromisoformat(data_dict['updated_at']),
                metadata=data_dict['metadata']
            )
            
            return invoice_data
            
        except Exception as exc:
            self.logger.error(f"Invoice retrieval failed: {exc}")
            return None


# Global instance
invoice_manager = InvoiceManager()


# Export main classes
__all__ = [
    'InvoiceManager',
    'InvoiceData',
    'InvoiceLineItem',
    'InvoiceRecipient',
    'InvoiceStatus',
    'InvoiceType',
    'DeliveryMethod',
    'InvoiceTemplateEngine',
    'InvoicePDFGenerator',
    'InvoiceStorageManager',
    'InvoiceEmailDelivery',
    'invoice_manager'
]
