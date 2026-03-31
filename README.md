\documentclass[10pt,conference,letterpaper]{IEEEtran}
\IEEEoverridecommandlockouts

% 1. Fix Column Gutter (most important - was 0.166in, needs >= 0.2in)
\setlength{\columnsep}{0.24in}

% 2. Fix Top and Right Margins
\newcommand{\CLASSINPUTtoptextmargin}{0.75in}
\newcommand{\CLASSINPUTbottomtextmargin}{1.0in}

% 3. Remove Bookmarks (IEEE does not allow them)
\usepackage[bookmarks=false,
            hypertexnames=false,
            colorlinks=true,
            linkcolor=blue,
            citecolor=blue,
            urlcolor=blue]{hyperref}

% 4. Improve Font Embedding
\usepackage{times}        % Helps with standard IEEE font embedding

% ====================== YOUR EXISTING PACKAGES (Cleaned) ======================

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{url}

% ====================== PROBLEMATIC PACKAGES (Comment or Remove) ======================

% REMOVE or COMMENT these - they often cause margin/gutter/font issues:
%\usepackage{microtype}          % <-- Often breaks margins & gutter. Comment this out!
%\usepackage{lineno}             % Line numbers usually not allowed in final submission

% Other commented packages you had - keep them commented unless you really need them:
% \usepackage{multirow}
% \usepackage{amsthm}
% \usepackage{mathrsfs}
% \usepackage[title]{appendix}
% \usepackage{manyfoot}
% \usepackage{algorithmicx}
% \usepackage{algpseudocode}
% \usepackage{listings}
% \usepackage{natbib}
% \usepackage{rotating}
% \usepackage{caption}
% \usepackage{array}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{}

\author{
\IEEEauthorblockN{[Author Names Anonymised for Review]}
\IEEEauthorblockA{[Institution Anonymised for Review]}
}

\maketitle

% ============================================================
\begin{abstract}
% ... your abstract text here ...
\end{abstract}

% Rest of your paper (introduction, sections, etc.)

\end{document}





pdf	sidemargins	The right margin is 0.56 in on page 5 (widths: 7.14; 7.14; 7.14; 7.15; 7.26; 7.14 in), which is below the required margin of 0.57 in for letter-sized paper.	-
pdf	topmargins	The top margin is 0.65 in on page 4, which is below the required margin of 0.7 in.	-
pdf	gutter	The gutter between columns is 0.166 inches wide (on page 3), but should be at least 0.2 inches.	-
pdf	notembedded	One or more fonts are not embedded. (FAQ 109)	-
pdf	bookmarks	Bookmarks are not allowed. (FAQ 115)	-
Could you please check the format of your submission and resubmit the paper to within 24 hours?



\documentclass[10pt,conference,letterpaper]{IEEEtran}
\IEEEoverridecommandlockouts

% ====================== STRONGER FIXES FOR REMAINING ERRORS ======================

% 1. Force larger top margin (critical for page 4 error)
\newcommand{\CLASSINPUTtoptextmargin}{0.80in}   % Increased from 0.75in
\newcommand{\CLASSINPUTbottomtextmargin}{1.0in}

% 2. Gutter (keep safe value)
\setlength{\columnsep}{0.24in}

% 3. Bookmarks disabled + hyperref (safe settings)
\usepackage[bookmarks=false,
            hypertexnames=false,
            colorlinks=true,
            linkcolor=blue,
            citecolor=blue,
            urlcolor=blue]{hyperref}

% 4. Font settings - Better for embedding Times-like fonts
\usepackage{times}
\usepackage{mathptmx}   % Better math + text font compatibility

% ====================== YOUR PACKAGES (Cleaned) ======================

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{url}

% ====================== IMPORTANT: COMMENT THESE ======================
% These often cause margin or font problems
%\usepackage{microtype}   % <-- Comment this out (very common cause of issues)
%\usepackage{lineno}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

Dear Mr. Mohan:

When processing your IEEE Globecom 2026 SAC - EH paper #1571269697, entitled "SpecFusion-SSL: Spectrum Fusion Self-Supervised Learning for PPG-Based Heart Rate and Blood Pressure Monitoring", we found one or more manuscript problems:

pdf	notembedded	The font TimesNewRomanPS-BoldMT is not embedded in the file. (FAQ 109)	-
pdf	topmargins	The top margin is 0.65 in on page 4, which is below the required margin of 0.7 in.	-
Could you please check the format of your submission and resubmit the paper to within 24 hours?


