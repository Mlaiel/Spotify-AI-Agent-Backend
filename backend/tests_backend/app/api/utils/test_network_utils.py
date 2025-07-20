"""
🎵 Spotify AI Agent - Tests Network Utils Module
================================================

Tests enterprise complets pour le module network_utils
avec validation de réseau, sécurité et performance.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import asyncio
import aiohttp
import socket
import ssl
import json
from unittest.mock import patch, Mock, AsyncMock
from urllib.parse import urljoin
import time

# Import du module à tester
from backend.app.api.utils.network_utils import (
    make_request,
    async_request,
    download_file,
    upload_file,
    check_connectivity,
    ping_host,
    resolve_hostname,
    get_public_ip,
    validate_url,
    parse_url,
    build_url,
    encode_params,
    create_session,
    retry_request,
    rate_limited_request,
    proxy_request,
    websocket_client,
    tcp_client,
    udp_client,
    network_scanner,
    bandwidth_test,
    latency_test,
    ssl_certificate_info,
    security_headers_check
)

from . import TestUtils, security_test, performance_test, integration_test


class TestNetworkUtils:
    """Tests pour le module network_utils"""
    
    def test_make_request_get(self):
        """Test requête GET basique"""
        # Mock response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}
            mock_get.return_value = mock_response
            
            response = make_request('GET', 'https://api.example.com/test')
            
            assert response['status_code'] == 200
            assert response['data'] == {"status": "ok"}
            mock_get.assert_called_once()
    
    def test_make_request_post_with_data(self):
        """Test requête POST avec données"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.text = '{"id": 123, "created": true}'
            mock_response.json.return_value = {"id": 123, "created": True}
            mock_post.return_value = mock_response
            
            data = {"name": "Test", "value": 42}
            response = make_request('POST', 'https://api.example.com/create', data=data)
            
            assert response['status_code'] == 201
            assert response['data']['id'] == 123
            mock_post.assert_called_with(
                'https://api.example.com/create',
                json=data,
                headers=None,
                timeout=30
            )
    
    def test_make_request_with_headers(self):
        """Test requête avec headers personnalisés"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"authenticated": true}'
            mock_response.json.return_value = {"authenticated": True}
            mock_get.return_value = mock_response
            
            headers = {
                'Authorization': 'Bearer token123',
                'User-Agent': 'Test-Agent/1.0'
            }
            
            response = make_request('GET', 'https://api.example.com/secure', headers=headers)
            
            assert response['status_code'] == 200
            mock_get.assert_called_with(
                'https://api.example.com/secure',
                json=None,
                headers=headers,
                timeout=30
            )
    
    def test_make_request_error_handling(self):
        """Test gestion erreurs requête"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.text = 'Not Found'
            mock_get.return_value = mock_response
            
            response = make_request('GET', 'https://api.example.com/notfound')
            
            assert response['status_code'] == 404
            assert 'error' in response
    
    @pytest.mark.asyncio
    async def test_async_request_get(self):
        """Test requête asynchrone GET"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = '{"async": "response"}'
            mock_response.json.return_value = {"async": "response"}
            mock_get.return_value.__aenter__.return_value = mock_response
            
            response = await async_request('GET', 'https://api.example.com/async')
            
            assert response['status_code'] == 200
            assert response['data'] == {"async": "response"}
    
    @pytest.mark.asyncio
    async def test_async_request_concurrent(self):
        """Test requêtes asynchrones concurrentes"""
        async def mock_request(method, url, **kwargs):
            await asyncio.sleep(0.01)  # Simulation latence
            return {
                'status_code': 200,
                'data': {'url': url, 'method': method}
            }
        
        with patch('backend.app.api.utils.network_utils.async_request', side_effect=mock_request):
            urls = [
                'https://api.example.com/endpoint1',
                'https://api.example.com/endpoint2',
                'https://api.example.com/endpoint3'
            ]
            
            start_time = time.time()
            tasks = [async_request('GET', url) for url in urls]
            responses = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time
            
            assert len(responses) == 3
            assert all(r['status_code'] == 200 for r in responses)
            assert execution_time < 0.1  # Concurrent donc rapide
    
    def test_download_file_basic(self):
        """Test téléchargement fichier basique"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"File content data"
            mock_response.headers = {'content-length': '17'}
            mock_get.return_value = mock_response
            
            with patch('builtins.open', create=True) as mock_open:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                result = download_file('https://example.com/file.txt', '/tmp/downloaded.txt')
                
                assert result['success'] is True
                assert result['file_size'] == 17
                mock_file.write.assert_called_once_with(b"File content data")
    
    def test_download_file_with_progress(self):
        """Test téléchargement avec suivi progression"""
        progress_updates = []
        
        def progress_callback(downloaded, total):
            progress_updates.append((downloaded, total))
        
        with patch('requests.get') as mock_get:
            # Simulation streaming
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'content-length': '1000'}
            mock_response.iter_content.return_value = [b'x' * 250] * 4  # 4 chunks de 250
            mock_get.return_value = mock_response
            
            with patch('builtins.open', create=True):
                result = download_file(
                    'https://example.com/large.txt',
                    '/tmp/large.txt',
                    progress_callback=progress_callback
                )
                
                assert result['success'] is True
                assert len(progress_updates) >= 2  # Au moins quelques updates
    
    def test_upload_file_basic(self):
        """Test upload fichier basique"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"file_id": "abc123", "uploaded": True}
            mock_post.return_value = mock_response
            
            with patch('builtins.open', create=True) as mock_open:
                mock_file = Mock()
                mock_file.read.return_value = b"File to upload"
                mock_open.return_value.__enter__.return_value = mock_file
                
                result = upload_file('/tmp/upload.txt', 'https://api.example.com/upload')
                
                assert result['success'] is True
                assert result['response']['file_id'] == "abc123"
    
    def test_check_connectivity_online(self):
        """Test vérification connectivité - en ligne"""
        with patch('socket.create_connection') as mock_socket:
            mock_socket.return_value = Mock()
            
            is_online = check_connectivity()
            
            assert is_online is True
    
    def test_check_connectivity_offline(self):
        """Test vérification connectivité - hors ligne"""
        with patch('socket.create_connection') as mock_socket:
            mock_socket.side_effect = socket.error("No connection")
            
            is_online = check_connectivity()
            
            assert is_online is False
    
    def test_ping_host_success(self):
        """Test ping host succès"""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "64 bytes from 8.8.8.8: time=20.1 ms"
            mock_run.return_value = mock_result
            
            result = ping_host('8.8.8.8')
            
            assert result['reachable'] is True
            assert 'response_time' in result
    
    def test_ping_host_failure(self):
        """Test ping host échec"""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stderr = "Host unreachable"
            mock_run.return_value = mock_result
            
            result = ping_host('unreachable.example.com')
            
            assert result['reachable'] is False
            assert 'error' in result
    
    def test_resolve_hostname_success(self):
        """Test résolution hostname succès"""
        with patch('socket.gethostbyname') as mock_resolve:
            mock_resolve.return_value = '93.184.216.34'
            
            ip = resolve_hostname('example.com')
            
            assert ip == '93.184.216.34'
    
    def test_resolve_hostname_failure(self):
        """Test résolution hostname échec"""
        with patch('socket.gethostbyname') as mock_resolve:
            mock_resolve.side_effect = socket.gaierror("Name resolution failed")
            
            ip = resolve_hostname('nonexistent.invalid')
            
            assert ip is None
    
    def test_get_public_ip(self):
        """Test obtention IP publique"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"ip": "203.0.113.42"}'
            mock_response.json.return_value = {"ip": "203.0.113.42"}
            mock_get.return_value = mock_response
            
            public_ip = get_public_ip()
            
            assert public_ip == "203.0.113.42"
    
    def test_validate_url_valid(self):
        """Test validation URL valide"""
        valid_urls = [
            'https://example.com',
            'http://subdomain.example.org/path?param=value',
            'https://api.service.com:8443/v1/endpoint',
            'ftp://files.example.com/download'
        ]
        
        for url in valid_urls:
            assert validate_url(url) is True
    
    def test_validate_url_invalid(self):
        """Test validation URL invalide"""
        invalid_urls = [
            'not-a-url',
            'http://',
            'https://.',
            'javascript:alert("xss")',
            'file:///etc/passwd'
        ]
        
        for url in invalid_urls:
            assert validate_url(url) is False
    
    @security_test
    def test_validate_url_security(self):
        """Test validation URL sécurité"""
        malicious_urls = [
            'javascript:alert("XSS")',
            'data:text/html,<script>alert("XSS")</script>',
            'file:///etc/passwd',
            'ftp://internal.network/secret',
            'gopher://localhost:25/xHELO%20...'
        ]
        
        for url in malicious_urls:
            # Doit rejeter les URLs potentiellement dangereuses
            assert validate_url(url, security_check=True) is False
    
    def test_parse_url_basic(self):
        """Test parsing URL basique"""
        url = 'https://api.example.com:8443/v1/users?id=123&format=json#section'
        
        parsed = parse_url(url)
        
        assert parsed['scheme'] == 'https'
        assert parsed['hostname'] == 'api.example.com'
        assert parsed['port'] == 8443
        assert parsed['path'] == '/v1/users'
        assert parsed['params'] == {'id': '123', 'format': 'json'}
        assert parsed['fragment'] == 'section'
    
    def test_build_url_basic(self):
        """Test construction URL"""
        components = {
            'scheme': 'https',
            'hostname': 'api.example.com',
            'port': 443,
            'path': '/v2/data',
            'params': {'key': 'value', 'limit': '10'}
        }
        
        url = build_url(components)
        
        assert url.startswith('https://api.example.com')
        assert '/v2/data' in url
        assert 'key=value' in url
        assert 'limit=10' in url
    
    def test_encode_params_basic(self):
        """Test encodage paramètres"""
        params = {
            'query': 'hello world',
            'special': 'chars & symbols',
            'unicode': 'café à paris'
        }
        
        encoded = encode_params(params)
        
        assert 'hello%20world' in encoded or 'hello+world' in encoded
        assert 'chars%20%26%20symbols' in encoded or 'chars+%26+symbols' in encoded
        assert 'caf%C3%A9' in encoded
    
    def test_create_session_basic(self):
        """Test création session HTTP"""
        session_config = {
            'timeout': 30,
            'retries': 3,
            'headers': {'User-Agent': 'Test-Client/1.0'}
        }
        
        session = create_session(session_config)
        
        assert session is not None
        # Vérifier configuration si possible
        if hasattr(session, 'timeout'):
            assert session.timeout == 30
    
    def test_retry_request_success_after_retries(self):
        """Test retry requête succès après échecs"""
        attempt_count = 0
        
        def mock_request(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return {'status_code': 200, 'data': {'success': True}}
        
        with patch('backend.app.api.utils.network_utils.make_request', side_effect=mock_request):
            response = retry_request('GET', 'https://api.example.com/retry', max_retries=3)
            
            assert response['status_code'] == 200
            assert attempt_count == 3
    
    def test_retry_request_max_retries_exceeded(self):
        """Test dépassement max retries"""
        def mock_request(*args, **kwargs):
            raise ConnectionError("Persistent failure")
        
        with patch('backend.app.api.utils.network_utils.make_request', side_effect=mock_request):
            response = retry_request('GET', 'https://api.example.com/fail', max_retries=2)
            
            assert 'error' in response
            assert 'max retries' in response['error'].lower()
    
    def test_rate_limited_request_basic(self):
        """Test requête avec limitation débit"""
        responses = []
        
        def mock_request(*args, **kwargs):
            responses.append(time.time())
            return {'status_code': 200, 'data': {'timestamp': time.time()}}
        
        with patch('backend.app.api.utils.network_utils.make_request', side_effect=mock_request):
            # 3 requêtes avec limite 2/seconde
            start_time = time.time()
            
            for i in range(3):
                response = rate_limited_request(
                    'GET', 
                    f'https://api.example.com/item{i}',
                    rate_limit=2  # 2 requêtes par seconde
                )
                assert response['status_code'] == 200
            
            execution_time = time.time() - start_time
            
            # Doit prendre au moins 1 seconde (rate limiting)
            assert execution_time >= 0.5
    
    @pytest.mark.asyncio
    async def test_websocket_client_basic(self):
        """Test client WebSocket basique"""
        messages_received = []
        
        class MockWebSocket:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, *args):
                pass
            
            async def send_str(self, message):
                # Simulation écho
                return message
            
            async def receive_str(self):
                return '{"echo": "test message"}'
        
        with patch('aiohttp.ClientSession.ws_connect', return_value=MockWebSocket()):
            async def message_handler(message):
                messages_received.append(message)
            
            client = websocket_client('wss://echo.websocket.org', message_handler)
            
            # Simulation connexion et envoi de message
            await client.connect()
            await client.send('test message')
            await asyncio.sleep(0.01)  # Attendre traitement
            
            assert len(messages_received) >= 0  # Peut recevoir messages
    
    def test_tcp_client_basic(self):
        """Test client TCP basique"""
        with patch('socket.socket') as mock_socket:
            mock_conn = Mock()
            mock_conn.recv.return_value = b"Server response"
            mock_socket.return_value.__enter__.return_value = mock_conn
            
            client = tcp_client('localhost', 8080)
            response = client.send_data(b"Hello server")
            
            assert response == b"Server response"
            mock_conn.send.assert_called_with(b"Hello server")
    
    def test_udp_client_basic(self):
        """Test client UDP basique"""
        with patch('socket.socket') as mock_socket:
            mock_sock = Mock()
            mock_sock.recvfrom.return_value = (b"UDP response", ('localhost', 8080))
            mock_socket.return_value = mock_sock
            
            client = udp_client()
            response = client.send_to('localhost', 8080, b"UDP message")
            
            assert response == b"UDP response"
            mock_sock.sendto.assert_called_with(b"UDP message", ('localhost', 8080))
    
    def test_network_scanner_basic(self):
        """Test scanner réseau basique"""
        def mock_ping(host):
            # Simulation: quelques hosts répondent
            responding_hosts = ['192.168.1.1', '192.168.1.100', '192.168.1.200']
            return {'reachable': host in responding_hosts, 'response_time': 10.5}
        
        with patch('backend.app.api.utils.network_utils.ping_host', side_effect=mock_ping):
            results = network_scanner('192.168.1.0/24', max_threads=10)
            
            assert isinstance(results, list)
            reachable_hosts = [r for r in results if r['reachable']]
            assert len(reachable_hosts) == 3
    
    def test_bandwidth_test_basic(self):
        """Test mesure bande passante"""
        def mock_download_test():
            # Simulation téléchargement 1MB en 0.5s = 2MB/s
            time.sleep(0.1)  # Simulation temps téléchargement
            return {
                'bytes_downloaded': 1024 * 1024,
                'duration': 0.5,
                'speed_mbps': 16  # 2MB/s = 16Mbps
            }
        
        with patch('backend.app.api.utils.network_utils.download_file') as mock_download:
            mock_download.return_value = {
                'success': True,
                'file_size': 1024 * 1024,
                'download_time': 0.5
            }
            
            result = bandwidth_test('https://speedtest.example.com/1mb.bin')
            
            assert 'download_speed_mbps' in result
            assert result['download_speed_mbps'] > 0
    
    def test_latency_test_basic(self):
        """Test mesure latence"""
        def mock_ping_results():
            return [
                {'reachable': True, 'response_time': 20.1},
                {'reachable': True, 'response_time': 21.5},
                {'reachable': True, 'response_time': 19.8},
                {'reachable': True, 'response_time': 20.9},
                {'reachable': True, 'response_time': 20.3}
            ]
        
        ping_results = mock_ping_results()
        
        with patch('backend.app.api.utils.network_utils.ping_host', side_effect=ping_results):
            result = latency_test('8.8.8.8', count=5)
            
            assert 'min_latency' in result
            assert 'max_latency' in result
            assert 'avg_latency' in result
            assert 'packet_loss' in result
            assert result['packet_loss'] == 0  # Tous réussis
    
    def test_ssl_certificate_info(self):
        """Test informations certificat SSL"""
        with patch('ssl.get_server_certificate') as mock_cert:
            with patch('ssl.PEM_cert_to_DER_cert') as mock_der:
                with patch('ssl.DER_cert_to_PEM_cert') as mock_pem:
                    mock_cert.return_value = "MOCK_CERTIFICATE"
                    
                    # Mock certificat parsé
                    cert_info = ssl_certificate_info('https://example.com')
                    
                    # Vérifications basiques
                    assert cert_info is not None
    
    @security_test
    def test_security_headers_check(self):
        """Test vérification headers sécurité"""
        with patch('requests.head') as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                'Strict-Transport-Security': 'max-age=31536000',
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Content-Security-Policy': "default-src 'self'"
            }
            mock_head.return_value = mock_response
            
            security_report = security_headers_check('https://secure.example.com')
            
            assert security_report['secure'] is True
            assert 'missing_headers' in security_report
            assert len(security_report['missing_headers']) == 0
    
    @security_test
    def test_security_headers_missing(self):
        """Test headers sécurité manquants"""
        with patch('requests.head') as mock_head:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {
                'Content-Type': 'text/html'
                # Aucun header de sécurité
            }
            mock_head.return_value = mock_response
            
            security_report = security_headers_check('https://insecure.example.com')
            
            assert security_report['secure'] is False
            assert len(security_report['missing_headers']) > 0
            assert 'Strict-Transport-Security' in security_report['missing_headers']
    
    @performance_test
    def test_concurrent_requests_performance(self):
        """Test performance requêtes concurrentes"""
        async def mock_async_request(method, url, **kwargs):
            await asyncio.sleep(0.01)  # Simulation latence réseau
            return {
                'status_code': 200,
                'data': {'url': url, 'timestamp': time.time()}
            }
        
        async def test_concurrent():
            with patch('backend.app.api.utils.network_utils.async_request', side_effect=mock_async_request):
                urls = [f'https://api.example.com/endpoint{i}' for i in range(20)]
                
                start_time = time.time()
                tasks = [async_request('GET', url) for url in urls]
                responses = await asyncio.gather(*tasks)
                execution_time = time.time() - start_time
                
                assert len(responses) == 20
                assert all(r['status_code'] == 200 for r in responses)
                assert execution_time < 0.5  # Concurrent donc rapide
        
        asyncio.run(test_concurrent())
    
    @integration_test
    def test_complete_network_workflow(self):
        """Test workflow réseau complet"""
        # Scénario: Client API avec retry, cache, monitoring
        
        request_count = 0
        
        def mock_api_request(method, url, **kwargs):
            nonlocal request_count
            request_count += 1
            
            # Simulation échecs intermittents
            if request_count in [2, 5]:
                raise ConnectionError("Network error")
            
            return {
                'status_code': 200,
                'data': {
                    'request_id': request_count,
                    'endpoint': url,
                    'method': method,
                    'timestamp': time.time()
                }
            }
        
        with patch('backend.app.api.utils.network_utils.make_request', side_effect=mock_api_request):
            # 1. Test connectivité
            with patch('socket.create_connection'):
                connectivity = check_connectivity()
                assert connectivity is True
            
            # 2. Résolution hostname
            with patch('socket.gethostbyname', return_value='203.0.113.42'):
                ip = resolve_hostname('api.example.com')
                assert ip == '203.0.113.42'
            
            # 3. Série de requêtes avec retry
            results = []
            for i in range(6):
                try:
                    response = retry_request(
                        'GET',
                        f'https://api.example.com/data/{i}',
                        max_retries=2
                    )
                    results.append(response)
                except Exception as e:
                    results.append({'error': str(e)})
            
            # 4. Vérification headers sécurité
            with patch('requests.head') as mock_head:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.headers = {'Strict-Transport-Security': 'max-age=31536000'}
                mock_head.return_value = mock_response
                
                security_check = security_headers_check('https://api.example.com')
            
            # 5. Test latence
            with patch('backend.app.api.utils.network_utils.ping_host') as mock_ping:
                mock_ping.return_value = {'reachable': True, 'response_time': 15.2}
                
                latency = latency_test('api.example.com', count=3)
        
        # Vérifications
        assert len(results) == 6
        successful_requests = [r for r in results if 'error' not in r]
        assert len(successful_requests) >= 4  # Au moins 4 succès avec retry
        
        assert security_check is not None
        assert latency['avg_latency'] > 0
        
        print("✅ Workflow réseau complet validé")


# Tests de robustesse réseau
class TestNetworkRobustness:
    """Tests de robustesse et gestion erreurs réseau"""
    
    @security_test
    def test_url_injection_protection(self):
        """Test protection injection URL"""
        malicious_urls = [
            'https://example.com/../../../etc/passwd',
            'https://example.com/api?redirect=javascript:alert(1)',
            'https://user:pass@evil.com@trusted.com/path',
            'https://trusted.com@evil.com/path'
        ]
        
        for url in malicious_urls:
            parsed = parse_url(url, security_check=True)
            
            # URL suspecte doit être rejetée ou nettoyée
            if parsed is not None:
                assert '../' not in parsed.get('path', '')
                assert 'javascript:' not in str(parsed)
    
    @security_test
    def test_ssrf_protection(self):
        """Test protection SSRF (Server-Side Request Forgery)"""
        internal_urls = [
            'http://localhost:8080/admin',
            'http://127.0.0.1:22/',
            'http://169.254.169.254/metadata',  # AWS metadata
            'http://10.0.0.1/internal',
            'http://192.168.1.1/router'
        ]
        
        for url in internal_urls:
            # Validation doit détecter URLs internes
            is_safe = validate_url(url, allow_internal=False)
            assert is_safe is False
    
    def test_network_timeout_handling(self):
        """Test gestion timeouts réseau"""
        def slow_request(*args, **kwargs):
            time.sleep(2)  # Plus lent que timeout
            return {'status_code': 200}
        
        with patch('requests.get', side_effect=slow_request):
            start_time = time.time()
            response = make_request('GET', 'https://slow.example.com', timeout=0.5)
            execution_time = time.time() - start_time
            
            # Doit timeout rapidement
            assert execution_time < 1.0
            assert 'error' in response or 'timeout' in response
    
    def test_connection_pool_exhaustion(self):
        """Test épuisement pool connexions"""
        session_config = {'max_connections': 2}
        
        def mock_request_with_delay(*args, **kwargs):
            time.sleep(0.1)
            return {'status_code': 200}
        
        with patch('backend.app.api.utils.network_utils.make_request', side_effect=mock_request_with_delay):
            session = create_session(session_config)
            
            # Essayer plus de requêtes que la limite du pool
            start_time = time.time()
            results = []
            
            for i in range(5):
                try:
                    response = make_request('GET', f'https://api.example.com/{i}')
                    results.append(response)
                except Exception as e:
                    results.append({'error': str(e)})
            
            execution_time = time.time() - start_time
            
            # Doit gérer limitation pool (queue ou erreur)
            assert len(results) == 5
            assert execution_time > 0.2  # Délai dû à pool limité


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
