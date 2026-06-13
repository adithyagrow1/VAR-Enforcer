# Script to create the complete index.html file
import os

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VAR ENFORCER - FIFA 2026</title>
    
    <!-- Google Fonts - Space Grotesk -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;700;900&display=swap" rel="stylesheet">
    
    <!-- GSAP -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js"></script>
    
    <style>
        /* CSS content will be added via script execution */
    </style>
</head>
<body>
    <h1>VAR ENFORCER</h1>
    <p>Placeholder - Full content will be generated</p>
</body>
</html>"""

# Write to file
with open('app/static/index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML file created successfully!")
print(f"File size: {len(html_content)} characters")

# Made with Bob
