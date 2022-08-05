% Based on Roland Bless's wissdoc.cls
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{wissdoc2}[2015/06/01 Lorenz Diener - long form scientific document]
\LoadClass[a4paper,12pt,titlepage]{book}

% English and German are possible
\usepackage[english,ngerman]{babel}

% Font info
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage[utf8]{inputenc} 

% Want graphics includes
\usepackage[final]{graphicx}
\usepackage{makeidx}
\usepackage[a4paper]{geometry}
\usepackage{fancyhdr}

% Pretty tables
\usepackage{array}
\usepackage{colortbl}
\usepackage{longtable}
\usepackage{booktabs}

% URLs
\usepackage[T1]{url}
\usepackage{hyperref}

% Typography improvements
\usepackage{microtype}

% Annotating
\usepackage{marginnote}

% Nicer bibliography style
\usepackage{natbib}

% Floats and subfigures
\usepackage{float} 
\usepackage{subfigure}
%\usepackage{subcaption}
% Colours
\usepackage{color} 
\usepackage{xcolor}

% Enumeration style control
\usepackage{enumitem}
\usepackage{colortbl}

% Allow for PDF inclusion
\usepackage{pstricks}

% Better control over captions and floats
\usepackage{tabularx} 
\usepackage{caption}

% Spacing tools
\usepackage{xspace}

% Mathematics
\usepackage{amssymb,amsmath,bm}
\usepackage{textcomp}

% Default numbering and TOC down to subsubsections
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

% Page settings
\geometry{left=3cm,right=3cm,top=23mm,bottom=25mm,head=14.5pt}

% Paragraph settings
\parindent 0cm
\parskip1.5ex plus0.5ex minus0.5ex

% Fonts for captions et cetera
\newcommand{\headfont}{\slshape}
\captionsetup{font={small},width=0.8\textwidth}
\newcommand{\chapterheadfont}{}

% Page styling
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}
\fancyhf{} % clear all header and footer fields
\fancyhead[LE,RO]{{\headfont\thepage}} % left/right header for even/odd pages
\fancyhead[LO]{\headfont\nouppercase{\rightmark}} % header for left side (odd)
\fancyhead[RE]{\headfont\nouppercase{\leftmark}} % right header for even pages
\renewcommand{\headrulewidth}{0.5pt} % head rule
\renewcommand{\footrulewidth}{0pt} % no foot rule

% Plain style
\fancypagestyle{plain}{%
\fancyhf{} % clear all header and footer fields
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
}

% Headings styling
\renewcommand{\chaptername}{}

\renewcommand{\section}{
\@startsection{section}%
{1}
{0mm}
{1.5ex plus 1ex minus 1.2ex}
{0.5ex plus 0.5ex minus 0.5ex}
{\chapterheadfont\Large\bfseries}
}
\renewcommand{\subsection}{
\@startsection{subsection}
{2}
{0mm}
{1ex plus 1ex minus 1ex}
{0.3ex plus 0.3ex minus 0.3ex}
{\chapterheadfont\large\bfseries}
}
\renewcommand{\subsubsection}{
\@startsection{subsubsection}
{3}
{0mm}
{1ex plus 1ex minus 1ex}
{0.2ex plus 0.2ex minus 0.2ex}
{\chapterheadfont\normalsize\bfseries}
}
\renewcommand{\paragraph}{
\@startsection{paragraph}
{4}
{0mm}
{2ex plus 1ex minus 2ex}
{0.2ex plus 0.2ex minus 0.2ex}
{\chapterheadfont\normalsize\bfseries}
}

% Chapter headings styling
\newlength{\chapnolen}
\newlength{\chapparlen}
\newsavebox{\chapno}
\renewcommand{\@makechapterhead}[1]{
  \vspace*{0.2\textheight}
  \vskip 15\p@
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \savebox{\chapno}{\chapterheadfont\huge\bfseries \thechapter.}
        \settowidth{\chapnolen}{\usebox{\chapno}}
        \parbox[t]{\chapnolen}{\usebox{\chapno}}\nobreak\leavevmode
      \fi
    \fi
    \interlinepenalty\@MM
    \setlength{\chapparlen}{\textwidth}
    \addtolength{\chapparlen}{-1.0\chapnolen}
    \addtolength{\chapparlen}{-2ex}
    \leavevmode\nobreak
    \parbox[t]{\chapparlen}{\raggedright\chapterheadfont\huge 
      \bfseries #1\par\nobreak}
    \vskip 40\p@
  }}

% Same, but unnumbered
\renewcommand{\@makeschapterhead}[1]{
  \vspace*{50\p@}
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \chapterheadfont \huge \bfseries  #1\par\nobreak
    \vskip 40\p@
  }
}




% Abstract environment
\newenvironment{abstract}{
  \thispagestyle{empty}
  \null\vfil
  \@beginparpenalty\@lowpenalty
  \begin{otherlanguage}{english}
  \begin{center}
    \bfseries \abstractname
    \@endparpenalty\@M
  \end{center}
  \end{otherlanguage}}
{\par\vfil\null}

\newenvironment{abstractgerman}{
  \thispagestyle{empty}
  \null\vfil
  \@beginparpenalty\@lowpenalty
  \begin{otherlanguage}{ngerman}
  \begin{center}
    \bfseries \abstractname
    \@endparpenalty\@M
  \end{center}
  \end{otherlanguage}}
{\par\vfil\null}

% TOC
\let\oldtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{{\pdfbookmark{\contentsname}{\contentsname}
  \chapterheadfont\oldtableofcontents}}
\let\@olddottedtocline\@dottedtocline
\renewcommand{\@dottedtocline}[5]{\@olddottedtocline{#1}{#2}{#3}{#4}{
  \chapterheadfont #5}}

% Title page style
\def\titlehead#1{\def\@titlehead{#1}}
\def\titlefoot#1{\def\@titlefoot{#1}}
\def\@titlehead{}
\def\@titlefoot{}

\def\ps@title{
\headheight 15mm
\footskip   0cm
\addtolength{\textheight}{-15mm}
\let\@mkboth\@gobbletwo
  \def\@oddhead{\vbox{\hbox to\textwidth{\@titlehead}
                \vskip 1.5mm
                \hbox to\textwidth{\hrulefill}}}
  \def\@oddfoot{\vbox{\vskip -1mm\hbox to\textwidth{\hrulefill}\vskip 1mm
                \hbox to\textwidth{\@titlefoot}}}
  \let\@evenhead\@oddhead
  \let\@evenfoot\@oddfoot
}

\renewenvironment{titlepage}
{
  \let\oldheadheight\headheight
  \let\oldfootskip\footskip
  \let\oldtextheight\textheight
  
   \cleardoublepage
   \if@twocolumn
      \@restonecoltrue\onecolumn
   \else
      \@restonecolfalse\newpage
    \fi
    \thispagestyle{plain}
    \setcounter{page}\@ne
}
{\if@restonecol\twocolumn \else \newpage \fi
 \if@twoside\else
    \setcounter{page}\@ne
 \fi
 \let\headheight\oldheadheight
 \let\textheight\oldtextheight
 \let\footskip\oldfootskip
}

% Indes page style
\renewenvironment{theindex}
{\if@twocolumn
   \@restonecolfalse
 \else
   \@restonecoltrue
 \fi
 \columnseprule \z@
 \columnsep 35\p@
 \twocolumn[\@makeschapterhead{\indexname}]
 \@mkboth{\indexname}
         {\indexname}
 \addcontentsline{toc}{chapter}{\indexname}
 \thispagestyle{fancy}
 \flushbottom
 \parindent\z@
 \parskip\z@ \@plus .3\p@\relax
 \let\item\@idxitem
 \def\,{\relax\ifmmode\mskip\thinmuskip
              \else\hskip0.2em\ignorespaces\fi}
 \raggedright}

% List prettification
\setlist[description]{leftmargin=1cm,labelindent=1cm,itemsep=-0.05cm} 

% Windows needs this
% \usepackage{auto-pst-pdf}
