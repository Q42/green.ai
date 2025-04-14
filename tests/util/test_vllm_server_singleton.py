import pytest
from unittest.mock import patch, Mock, MagicMock
import subprocess
from benchmarq.utility import VLLMServerSingleton


class TestVLLMServerSingleton:
    def test_singleton_pattern(self):
        # Test that multiple instances are the same object
        instance1 = VLLMServerSingleton()
        instance2 = VLLMServerSingleton()

        assert instance1 is instance2

    def test_initial_state(self):
        # Test initial state of the singleton
        server = VLLMServerSingleton()
        assert server._process is None

    @patch('subprocess.Popen')
    def test_start_server_success(self, mock_popen):
        # Set up mock for successful server start
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Starting server...",
            "INFO:     Application startup complete.",
            ""  # Empty string to exit the while loop
        ]
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        server = VLLMServerSingleton()
        server._process = None  # Reset for testing

        with patch('builtins.print') as mock_print:
            server.start_server("test-model")

            # Verify Popen was called correctly
            mock_popen.assert_called_once_with(
                ["vllm", "serve", "test-model"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Check that success message was printed
            mock_print.assert_called_with("Server started successfully.")

            # Check that process was stored
            assert server._process is mock_process

    @patch('subprocess.Popen')
    def test_start_server_failure(self, mock_popen):
        # Set up mock for failed server start
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Starting server...",
            "Error occurred",
            ""  # Empty string to exit the while loop
        ]
        mock_process.poll.return_value = 1  # Non-zero exit code indicating failure
        mock_popen.return_value = mock_process

        server = VLLMServerSingleton()
        server._process = None  # Reset for testing

        with patch('builtins.print') as mock_print:
            server.start_server("test-model")

            # Verify Popen was called
            mock_popen.assert_called_once()

            # Check that failure message was printed
            mock_print.assert_called_with("Failed to start the server.")

            # Check that process was reset to None
            assert server._process is None

    def test_start_server_already_running(self):
        # Test when server is already running
        server = VLLMServerSingleton()
        server._process = Mock()  # Simulate running process

        with patch('builtins.print') as mock_print:
            server.start_server("test-model")

            # Check that appropriate message was printed
            mock_print.assert_called_with("Server is already running.")

    def test_stop_server_running(self):
        # Test stopping server when it's running
        mock_process = Mock()
        
        server = VLLMServerSingleton()
        server._process = mock_process
        
        with patch('builtins.print') as mock_print:
            server.stop_server()
            
            # Verify process termination was attempted
            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called_once_with(timeout=5)

            # Check that process was set to None
            assert server._process is None

            # Check that appropriate message was printed
            mock_print.assert_called_with("Server stopped.")

    def test_stop_server_not_running(self):
        # Test stopping server when it's not running
        server = VLLMServerSingleton()
        server._process = None

        with patch('builtins.print') as mock_print:
            server.stop_server()

            # Check that appropriate message was printed
            mock_print.assert_called_with("No server is running.")

    def test_stop_server_timeout(self):
        # Test stopping server when termination times out
        mock_process = Mock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)

        server = VLLMServerSingleton()
        server._process = mock_process

        with patch('builtins.print') as mock_print:
            server.stop_server()

            # Verify termination was attempted
            mock_process.terminate.assert_called_once()
            # Verify kill was called after timeout
            mock_process.kill.assert_called_once()

            # Check that process was set to None
            assert server._process is None

            # Check that appropriate message was printed
            mock_print.assert_called_with("Server stopped.")
