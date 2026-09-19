import logging
import pytest
from src.utils.logger import get_logger

def test_logger_creation_and_format(capsys):
    # Create a logger for a named component
    logger = get_logger("test_component")
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_component"

    # Emit a log record
    logger.info("Test message for logger")
    
    # Capture output
    captured = capsys.readouterr()
    output = captured.out

    # Verify timestamp, log level, component name, and message exist
    # Output format is: 2026-09-19 15:55:00 | INFO     | test_component | Test message for logger
    assert "INFO" in output
    assert "test_component" in output
    assert "Test message for logger" in output
    
    # Naive timestamp check: contains a 4-digit year like 202
    assert "202" in output
    assert "|" in output

def test_no_duplicate_handlers(capsys):
    logger1 = get_logger("dup_test")
    logger2 = get_logger("dup_test")
    
    # Should be the identical logger object
    assert logger1 is logger2
    
    # Should only have a single handler attached
    assert len(logger1.handlers) == 1
    
    logger1.info("Should appear exactly once")
    
    captured = capsys.readouterr()
    output = captured.out
    
    # Should only contain the message exactly one time
    assert output.count("Should appear exactly once") == 1
