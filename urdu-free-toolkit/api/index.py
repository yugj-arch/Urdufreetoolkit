# -*- coding: utf-8 -*-
"""Vercel serverless entrypoint.

Vercel's Python runtime serves the module-level ``app`` WSGI callable, and the
rewrite in ``vercel.json`` sends every route here. The real app is one level up
in ``app.py`` — this just puts the project root on ``sys.path`` and re-exports
it, so the same code runs locally (``python app.py``) and on Vercel.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402,F401  — WSGI callable Vercel invokes
