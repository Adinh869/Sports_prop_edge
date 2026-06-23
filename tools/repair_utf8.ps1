$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$FixEnc = Join-Path $Root "tools\fix_enc.py"
$ImportTest = Join-Path $Root "tools\_import_test.py"
$Requirements = Join-Path $Root "requirements.txt"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false

Write-Host "[repair] project: $Root"
if (-not (Test-Path $Python)) {
    Write-Host "[repair] ERROR: missing $Python"
    exit 1
}

# Fix UTF-16 / null-byte files BEFORE pip (corrupt requirements.txt breaks pip)
Write-Host "[repair] scanning for UTF-16 / null-byte files..."
$fixed = 0
$patterns = @("*.py", "*.ps1", "*.csv", "*.bat", "*.mdc", "*.json", "*.toml", "*.md", "*.txt")
$skip = @(".venv", "__pycache__", ".git")
foreach ($pattern in $patterns) {
    Get-ChildItem -Path $Root -Recurse -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
        $parts = $_.FullName.Split([IO.Path]::DirectorySeparatorChar)
        if ($skip | Where-Object { $parts -contains $_ }) { return }
        $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
        if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
            $text = [System.Text.Encoding]::Unicode.GetString($bytes)
            [System.IO.File]::WriteAllText($_.FullName, $text, $utf8NoBom)
            Write-Host "  fixed (utf-16): $($_.FullName)"
            $fixed++
        }
        elseif ($bytes -contains 0) {
            $text = [System.Text.Encoding]::Unicode.GetString($bytes)
            [System.IO.File]::WriteAllText($_.FullName, $text, $utf8NoBom)
            Write-Host "  fixed (null bytes): $($_.FullName)"
            $fixed++
        }
    }
}
Write-Host "[repair] powershell pass fixed $fixed file(s)"

Write-Host "[repair] running fix_enc.py..."
& $Python $FixEnc
if ($LASTEXITCODE -ne 0) {
    Write-Host "[repair] fix_enc failed exit $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[repair] ensuring pip deps (beautifulsoup4, pyarrow, etc.)..."
& $Python -m pip install -q -r $Requirements
if ($LASTEXITCODE -ne 0) {
    Write-Host "[repair] pip install failed exit $LASTEXITCODE"
    exit $LASTEXITCODE
}

$env:PYTHONPATH = Join-Path $Root "src"
$body = "from sports_prop_edge.strategy.payouts import PayoutProfile`nfrom sports_prop_edge.strategy.card_builder import CardRules`nprint('imports ok')"
[System.IO.File]::WriteAllText($ImportTest, $body, $utf8NoBom)
Write-Host "[repair] testing imports..."
& $Python $ImportTest
$code = $LASTEXITCODE
Remove-Item $ImportTest -ErrorAction SilentlyContinue
if ($code -ne 0) {
    Write-Host "[repair] import test FAILED exit $code"
    exit $code
}
Write-Host "[repair] all good"
