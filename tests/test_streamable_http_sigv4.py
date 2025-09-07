"""Essential tests for streamable HTTP with SigV4 authentication."""

from unittest.mock import Mock, patch
from botocore.credentials import Credentials

from langchain_mcp_adapters.streamable_http_sigv4 import SigV4HTTPXAuth


def test_sigv4_auth_initialization():
    """Test SigV4HTTPXAuth initializes with correct credentials and region."""
    credentials = Credentials(access_key="test_key", secret_key="test_secret")

    auth = SigV4HTTPXAuth(credentials, "bedrock-agentcore", "us-east-1")

    assert auth.credentials == credentials
    assert auth.service == "bedrock-agentcore"
    assert auth.region == "us-east-1"


def test_auth_flow_removes_connection_header():
    """Test that connection header is removed during SigV4 signing."""
    credentials = Credentials(access_key="test_key", secret_key="test_secret")
    auth = SigV4HTTPXAuth(credentials, "bedrock-agentcore", "us-east-1")

    request = Mock()
    request.method = "POST"
    request.url = "https://example.com/api"
    request.content = b"{}"
    request.headers = {"connection": "keep-alive", "content-type": "application/json"}

    with patch.object(auth.signer, "add_auth"):
        with patch("botocore.awsrequest.AWSRequest") as mock_aws_request:
            list(auth.auth_flow(request))

            # Verify connection header was removed from AWS request
            call_args = mock_aws_request.call_args[1]
            assert "connection" not in call_args["headers"]
            assert "content-type" in call_args["headers"]
