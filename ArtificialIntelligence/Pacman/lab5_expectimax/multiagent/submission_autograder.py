#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))
