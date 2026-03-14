
import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';

const DEFAULT_TEXT =
  'The experienced chief surgeon successfully performed the complex operation. The nurse helped him and handed over the instruments. After three hours, the procedure was completed.\n\nThe chairman decided he would lead the project.';

// API-Call für Bias Detection (Klassifikation)
async function fetchBiasPredictions(contexts, texts) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ context_texts: contexts, target_texts: texts }),
  });
  return response.json();
}

// API-Call für LLM-Explanation
async function fetchLLMExplanation(context_before, flagged_sentence, context_after) {
  const response = await fetch('http://localhost:8000/explain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ context_before, flagged_sentence, context_after }),
  });
  return response.json();
}

// Hilfsfunktion: Finde alle Vorkommen von Problemtexten und markiere sie
const buildHighlightedSegments = (text, findings) => {
  const activeFindings = findings.filter((finding) => !finding.accepted);
  if (activeFindings.length === 0) {
    return [{ type: 'text', value: text }];
  }
  let segments = [];
  let cursor = 0;
  let sorted = [...activeFindings].sort((a, b) => text.indexOf(a.problematicText) - text.indexOf(b.problematicText));
  sorted.forEach((finding) => {
    const index = text.indexOf(finding.problematicText, cursor);
    if (index !== -1) {
      if (index > cursor) {
        segments.push({ type: 'text', value: text.slice(cursor, index) });
      }
      segments.push({ type: 'issue', value: finding.problematicText, finding });
      cursor = index + finding.problematicText.length;
    }
  });
  if (cursor < text.length) {
    segments.push({ type: 'text', value: text.slice(cursor) });
  }
  return segments;
};

export default function BiasEditor() {
  const [inputText, setInputText] = useState(DEFAULT_TEXT);
  const [processedText, setProcessedText] = useState('');
  const [findings, setFindings] = useState([]); // [{problematicText, biasType, explanation, rewriteSuggestion, accepted}]
  const [analysisDone, setAnalysisDone] = useState(false);
  const [activeHoverId, setActiveHoverId] = useState(null);
  const [lastFixedId, setLastFixedId] = useState(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });

  const openTimeoutRef = useRef(null);

  const clearHoverTimeouts = () => {
    if (openTimeoutRef.current) {
      clearTimeout(openTimeoutRef.current);
      openTimeoutRef.current = null;
    }
  };

  const updateHoverPosition = (event) => {
    const tooltipWidth = 380;
    const tooltipHeight = 240;
    const viewportPadding = 16;

    let nextX = event.clientX + 12;
    if (nextX + tooltipWidth + viewportPadding > window.innerWidth) {
      nextX = Math.max(viewportPadding, window.innerWidth - tooltipWidth - viewportPadding);
    }

    let nextY = event.clientY + 16;
    if (nextY + tooltipHeight + viewportPadding > window.innerHeight) {
      nextY = event.clientY - tooltipHeight - 16;
    }

    if (nextY < viewportPadding) {
      nextY = viewportPadding;
    }

    setHoverPosition({ x: nextX, y: nextY });
  };

  const handleHoverStart = (findingId, event) => {
    const isSameOpenFinding = activeHoverId === findingId;

    if (event && !isSameOpenFinding) {
      updateHoverPosition(event);
    }

    clearHoverTimeouts();

    if (isSameOpenFinding) {
      return;
    }

    openTimeoutRef.current = setTimeout(() => {
      setActiveHoverId(findingId);
    }, 90);
  };

  const closePopover = () => {
    clearHoverTimeouts();
    setActiveHoverId(null);
  };

  // Analyse-Logik: Text an Backend schicken, Findings setzen
  const processText = async () => {
    const trimmed = inputText.trim();
    const baseText = trimmed.length > 0 ? inputText : '';
    setProcessedText(baseText);
    setFindings([]);
    setAnalysisDone(false);
    setActiveHoverId(null);
    setLastFixedId(null);

    // Beispiel: Sätze splitten (hier sehr einfach, besser: Backend)
    const sentences = baseText.split(/(?<=[.!?])\s+/);
    let findingsArr = [];
    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i];
      // Kontext: vorheriger Satz, nachfolgender Satz
      const context_before = i > 0 ? sentences[i - 1] : '';
      const context_after = i < sentences.length - 1 ? sentences[i + 1] : '';
      // LLM-API aufrufen
      try {
        const result = await fetchLLMExplanation(context_before, sentence, context_after);
        if (result && result.bias_type && result.bias_type !== "") {
          findingsArr.push({
            id: `bias-${i}`,
            problematicText: sentence,
            biasType: result.bias_type,
            explanation: result.explanation,
            rewriteSuggestion: result.rewrite_suggestion,
            accepted: false,
          });
        }
      } catch (e) {
        // Fehler ignorieren, Satz überspringen
      }
    }
    setFindings(findingsArr);
    setAnalysisDone(true);
  };

  const handleAcceptSuggestion = (findingId) => {
    const selectedFinding = findings.find((finding) => finding.id === findingId);
    if (!selectedFinding) {
      return;
    }

    setProcessedText((currentText) =>
      currentText.replace(selectedFinding.problematicText, selectedFinding.rewriteSuggestion)
    );

    setFindings((currentFindings) =>
      currentFindings.map((finding) =>
        finding.id === findingId ? { ...finding, accepted: true } : finding
      )
    );

    setActiveHoverId(null);
    setLastFixedId(findingId);
  };

  useEffect(() => {
    return () => {
      clearHoverTimeouts();
    };
  }, []);

  const highlightedSegments = buildHighlightedSegments(processedText, findings);
  const openFinding = findings.find((finding) => finding.id === activeHoverId);
  const hasActiveFinding = findings.some((finding) => !finding.accepted);

  return (
    <div className="min-h-screen bg-slate-50 p-6 md:p-10">
      <div className="mx-auto max-w-4xl">
        <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-xl md:p-10">
          <div className="mb-6 flex items-center justify-between">
            <h1 className="text-lg font-semibold text-slate-900">Bias Scanner</h1>
            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
              Mock Prototype
            </span>
          </div>

          <div className="space-y-5">
            <div>
              <label htmlFor="bias-input" className="mb-2 block text-sm font-medium text-slate-700">
                Text Input
              </label>
              <textarea
                id="bias-input"
                value={inputText}
                onChange={(event) => setInputText(event.target.value)}
                placeholder="Paste or write your text here..."
                className="min-h-[180px] w-full rounded-xl border border-slate-200 bg-white p-4 text-sm leading-6 text-slate-800 shadow-sm outline-none transition focus:border-blue-300 focus:ring-2 focus:ring-blue-100"
              />
            </div>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={processText}
                className="inline-flex items-center justify-center rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white transition-all duration-200 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300"
              >
                Process
              </button>
            </div>

            {analysisDone && (
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-slate-900">Processed Result</h2>
                  <span className="text-xs text-slate-500">
                    {findings.filter((finding) => !finding.accepted).length} active issue(s)
                  </span>
                </div>

                <div className="relative rounded-xl border border-slate-200 bg-white p-4">
                  <p className="whitespace-pre-wrap text-[16px] leading-8 text-slate-800">
                    {highlightedSegments.map((segment, index) => {
                      if (segment.type === 'text') {
                        return <span key={`text-${index}`}>{segment.value}</span>;
                      }

                      return (
                        <span
                          key={segment.finding.id}
                          className="relative inline-block"
                          onMouseEnter={(event) => handleHoverStart(segment.finding.id, event)}
                        >
                          <span className="cursor-pointer rounded-sm border-b-2 border-red-300 bg-red-50 px-1 text-slate-900 transition-colors duration-200 hover:bg-red-100">
                            {segment.value}
                          </span>
                        </span>
                      );
                    })}
                  </p>

                  <div
                    className={`fixed z-50 w-[380px] transition-all duration-150 ${
                      openFinding
                        ? 'pointer-events-auto translate-y-0 opacity-100'
                        : 'pointer-events-none translate-y-1 opacity-0'
                    }`}
                    style={{ left: hoverPosition.x, top: hoverPosition.y }}
                    onMouseEnter={(event) => openFinding && handleHoverStart(openFinding.id, event)}
                    onMouseLeave={closePopover}
                  >
                    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-2xl">
                      <div className="mb-3 flex items-start gap-2">
                        <div className="mt-0.5 rounded-md bg-amber-100 p-1 text-amber-700">
                          <AlertTriangle size={14} />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-slate-900">
                            {openFinding?.biasType}
                          </p>
                          <p className="mt-1 text-sm leading-6 text-slate-600">
                            {openFinding?.explanation}
                          </p>
                        </div>
                      </div>

                      {openFinding && (
                        <>
                          <div className="rounded-xl border border-green-200 bg-green-50 p-3">
                            <p className="text-sm text-slate-500 line-through">
                              {openFinding.problematicText}
                            </p>
                            <p className="mt-2 text-sm font-medium text-green-800">
                              {openFinding.rewriteSuggestion}
                            </p>
                          </div>

                          <button
                            type="button"
                            onClick={() => handleAcceptSuggestion(openFinding.id)}
                            className="mt-3 inline-flex w-full items-center justify-center rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition-all duration-200 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300"
                          >
                            Accept Suggestion
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                {!hasActiveFinding && processedText.trim().length > 0 && (
                  <div className="mt-4 flex items-center gap-2 rounded-xl border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-800">
                    <CheckCircle2 size={16} />
                    <span>No active mocked bias issues found.</span>
                  </div>
                )}
              </div>
            )}

            {lastFixedId && (
              <div className="flex items-center gap-2 rounded-xl border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-800">
                <CheckCircle2 size={16} />
                <span>Suggestion applied successfully.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
