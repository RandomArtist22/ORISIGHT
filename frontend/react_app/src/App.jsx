import { useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ClipboardCheck,
  FileSearch,
  Microscope,
  RefreshCcw,
  ShieldPlus,
  Sparkles,
  Stethoscope,
  TestTube2,
  Upload
} from 'lucide-react'
import { analyzeCase, submitDoctorReview, triggerEmbed, triggerScrape } from './api'

const supportedDiseases = [
  'Oral Submucous Fibrosis',
  'Leukoplakia',
  'Erythroplakia',
  'Oral Lichen Planus',
  'Oral Squamous Cell Carcinoma'
]

const initialReviewDraft = {
  diagnosis: '',
  differential_diagnosis: [],
  risk_level: '',
  suggested_tests: [],
  treatment_plan: [],
  referral: '',
  confidence_score: '',
  notes: '',
  confirmed: false
}

export default function App() {
  const fileInputRef = useRef(null)
  const [imageFile, setImageFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [symptoms, setSymptoms] = useState('Burning sensation in buccal mucosa with persistent white patch for 6 weeks.')
  const [history, setHistory] = useState('Tobacco chewing for 8 years, areca nut use daily, occasional alcohol.')
  const [doctorMode, setDoctorMode] = useState(true)

  const [result, setResult] = useState(null)
  const [reviewDraft, setReviewDraft] = useState(initialReviewDraft)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [kbLoading, setKbLoading] = useState(false)
  const [kbMessage, setKbMessage] = useState('')
  const [expandedKnowledge, setExpandedKnowledge] = useState({})
  const [expandedImage, setExpandedImage] = useState(null)

  const riskClass = useMemo(() => {
    const level = result?.output?.risk_level?.toLowerCase() || ''
    if (level.includes('high')) return 'risk-pill risk-high'
    if (level.includes('moderate')) return 'risk-pill risk-moderate'
    return 'risk-pill risk-low'
  }, [result])

  async function handleAnalyze(event) {
    event.preventDefault()
    if (!imageFile) {
      setError('Please upload an oral cavity image.')
      return
    }

    setError('')
    setLoading(true)

    try {
      const data = await analyzeCase({ imageFile, symptoms, history, doctorMode })
      setResult(data)
      setReviewDraft({
        diagnosis: data.output.diagnosis,
        differential_diagnosis: data.output.differential_diagnosis,
        risk_level: data.output.risk_level,
        suggested_tests: data.output.suggested_tests,
        treatment_plan: data.output.treatment_plan,
        referral: data.output.referral,
        confidence_score: data.output.confidence_score,
        notes: '',
        confirmed: false
      })
      setExpandedKnowledge({})
      setExpandedImage(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleBootstrapKnowledge() {
    setKbLoading(true)
    setKbMessage('Collecting trusted oral disease references...')

    try {
      const scrape = await triggerScrape('oral potentially malignant disorders')
      setKbMessage(`Indexed ${scrape.saved_count} documents. Generating retrieval embeddings...`)
      const embed = await triggerEmbed(false)
      const chunks = embed?.medical_docs?.chunks_added || 0
      setKbMessage(`Knowledge base ready. ${chunks} new chunks added.`)
    } catch (err) {
      setKbMessage(`Knowledge refresh failed: ${err.message}`)
    } finally {
      setKbLoading(false)
    }
  }

  async function handleReviewSubmit(event) {
    event.preventDefault()
    if (!result?.case_id) return

    try {
      const payload = {
        ...reviewDraft,
        differential_diagnosis: toList(reviewDraft.differential_diagnosis),
        suggested_tests: toList(reviewDraft.suggested_tests),
        treatment_plan: toList(reviewDraft.treatment_plan)
      }

      const reviewed = await submitDoctorReview(result.case_id, payload)
      setResult(reviewed.report)
      setKbMessage('Doctor review saved successfully.')
    } catch (err) {
      setKbMessage(`Doctor review failed: ${err.message}`)
    }
  }

  function onSelectImage(file) {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setImageFile(file)
    setPreviewUrl(file ? URL.createObjectURL(file) : '')
  }

  function handleChooseFile() {
    fileInputRef.current?.click()
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragActive(false)
    const file = event.dataTransfer?.files?.[0] || null
    onSelectImage(file)
  }

  function handleDragOver(event) {
    event.preventDefault()
    if (!dragActive) setDragActive(true)
  }

  function handleDragLeave(event) {
    event.preventDefault()
    setDragActive(false)
  }

  function clearSelectedImage() {
    onSelectImage(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  function toggleKnowledgeChunk(index) {
    setExpandedKnowledge((prev) => ({ ...prev, [index]: !prev[index] }))
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-wrap">
          <ToothLogo />
          <div>
            <h1 className="brand-title">ORISIGHT</h1>
            <p className="brand-subtitle">Multimodal oral lesion decision support</p>
          </div>
        </div>

        <button type="button" onClick={handleBootstrapKnowledge} disabled={kbLoading} className="btn btn-secondary">
          <RefreshCcw className={`icon ${kbLoading ? 'spin' : ''}`} />
          {kbLoading ? 'Refreshing Knowledge...' : 'Refresh Knowledge Base'}
        </button>
      </header>

      <section className="notice warning">
        ORISIGHT is a hackathon MVP and not a real medical diagnostic device.
      </section>
      {kbMessage && <section className="notice info">{kbMessage}</section>}

      <main className="workspace-grid">
        <form className="card input-card" onSubmit={handleAnalyze}>
          <div className="section-head">
            <h2>New Patient Case</h2>
            <p>Upload an oral cavity image, then add symptoms and clinical history.</p>
          </div>

          <ol className="steps">
            <li><span>1</span> Upload image</li>
            <li><span>2</span> Add symptoms and history</li>
            <li><span>3</span> Run multimodal analysis</li>
          </ol>

          <label className="label" htmlFor="oral-image">Oral cavity image</label>
          <input
            ref={fileInputRef}
            id="oral-image"
            type="file"
            accept="image/*"
            onChange={(event) => onSelectImage(event.target.files?.[0] || null)}
            className="hidden-file-input"
          />

          <div
            className={`upload-dropzone ${dragActive ? 'drag-active' : ''}`}
            role="button"
            tabIndex={0}
            onClick={handleChooseFile}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                handleChooseFile()
              }
            }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Upload className="upload-icon" />
            <p className="upload-title">Drag and drop image here</p>
            <p className="muted">or tap to browse from your device</p>
            <button
              type="button"
              className="btn btn-secondary btn-small"
              onClick={(event) => {
                event.stopPropagation()
                handleChooseFile()
              }}
            >
              Choose file
            </button>
          </div>

          {imageFile && (
            <div className="file-pill">
              <div className="truncate-block">
                <p className="strong">{imageFile.name}</p>
                <p className="muted">{formatFileSize(imageFile.size)}</p>
              </div>
              <button type="button" className="text-btn" onClick={clearSelectedImage}>
                Remove
              </button>
            </div>
          )}

          {previewUrl && (
            <div className="preview-card">
              <img src={previewUrl} alt="Uploaded oral cavity" className="preview-image" />
            </div>
          )}

          <label className="label" htmlFor="symptoms">Symptoms</label>
          <textarea
            id="symptoms"
            rows={5}
            value={symptoms}
            onChange={(event) => setSymptoms(event.target.value)}
            className="input"
            placeholder="Pain, burning, ulceration, white/red patches, duration..."
          />

          <label className="label" htmlFor="history">Medical and habit history</label>
          <textarea
            id="history"
            rows={5}
            value={history}
            onChange={(event) => setHistory(event.target.value)}
            className="input"
            placeholder="Tobacco/areca/alcohol exposure, prior lesions, biopsy history..."
          />

          <label className="checkbox-row">
            <input type="checkbox" checked={doctorMode} onChange={(event) => setDoctorMode(event.target.checked)} />
            <span>Enable doctor review mode</span>
          </label>

          {error && (
            <p className="error-banner">
              <AlertTriangle className="icon" />
              {error}
            </p>
          )}

          <button type="submit" disabled={loading} className="btn btn-primary full">
            <Activity className="icon" />
            {loading ? 'Analyzing...' : 'Analyze Case'}
          </button>

          <div className="support-wrap">
            <p className="label">Supported diseases</p>
            <div className="chip-grid">
              {supportedDiseases.map((disease) => (
                <span key={disease} className="chip">{disease}</span>
              ))}
            </div>
          </div>
        </form>

        <div className="results-column">
          <section className="card">
            <div className="section-head">
              <h2>Analysis Report</h2>
              <p>Clear, structured output for quick triage review.</p>
            </div>

            {!result && (
              <div className="empty-block">
                <Sparkles className="icon-lg" />
                <p>Run analysis to generate probable diagnosis, differentials, investigations, and treatment guidance.</p>
              </div>
            )}

            {result && (
              <>
                <div className="diagnosis-row">
                  <div>
                    <p className="muted">Most likely diagnosis</p>
                    <h3 className="diagnosis-name">{result.output.diagnosis}</h3>
                  </div>
                  <span className={riskClass}>{result.output.risk_level} risk</span>
                </div>

                <div className="kpi-grid">
                  <article className="kpi-card">
                    <ClipboardCheck className="icon" />
                    <div>
                      <p className="muted">Confidence</p>
                      <p className="kpi-value">{result.output.confidence_score}</p>
                    </div>
                  </article>
                  <article className="kpi-card">
                    <Stethoscope className="icon" />
                    <div>
                      <p className="muted">Referral</p>
                      <p className="kpi-value small">{result.output.referral}</p>
                    </div>
                  </article>
                </div>

                <MetricList title="Differential Diagnoses" items={result.output.differential_diagnosis} icon={<FileSearch className="icon" />} />
                <MetricList title="Suggested Investigations" items={result.output.suggested_tests} icon={<TestTube2 className="icon" />} />
                <MetricList title="Treatment Plan" items={result.output.treatment_plan} icon={<ShieldPlus className="icon" />} />
              </>
            )}
          </section>

          {result && (
            <section className="card">
              <div className="section-head">
                <h2>Explainability</h2>
                <p>{result.explainability.image_caption}</p>
              </div>

              <div className="image-pair">
                <article className="image-card">
                  <p className="tiny-label">Uploaded image</p>
                  <img src={result.explainability.image_url} alt="Original lesion" className="result-image" />
                  <button
                    type="button"
                    className="text-btn image-expand-btn"
                    onClick={() =>
                      setExpandedImage({
                        src: result.explainability.image_url,
                        title: 'Uploaded Image'
                      })
                    }
                  >
                    Expand
                  </button>
                </article>
                <article className="image-card">
                  <p className="tiny-label">Lesion heatmap</p>
                  <img src={result.explainability.heatmap_url} alt="Heatmap overlay" className="result-image" />
                  <button
                    type="button"
                    className="text-btn image-expand-btn"
                    onClick={() =>
                      setExpandedImage({
                        src: result.explainability.heatmap_url,
                        title: 'Lesion Heatmap'
                      })
                    }
                  >
                    Expand
                  </button>
                </article>
              </div>

              <MetricList title="Extracted Risk Factors" items={result.explainability.risk_factors} icon={<Microscope className="icon" />} />

              <div className="split-block">
                <div>
                  <p className="tiny-label">Top Similar Cases</p>
                  {result.explainability.similar_cases.length === 0 && <p className="muted">No indexed similar cases yet.</p>}
                  <div className="small-grid">
                    {result.explainability.similar_cases.map((caseItem) => (
                      <article key={caseItem.case_id} className="small-card">
                        <p className="mono">{caseItem.case_id.slice(0, 8)}</p>
                        <p className="strong">{caseItem.diagnosis}</p>
                        <p className="muted">Similarity: {caseItem.similarity}</p>
                      </article>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="tiny-label">Retrieved Medical Knowledge</p>
                  {result.explainability.retrieved_knowledge.length === 0 && (
                    <p className="muted">Knowledge base is empty. Click "Refresh Knowledge Base".</p>
                  )}
                  <div className="knowledge-list">
                    {result.explainability.retrieved_knowledge.map((item, idx) => (
                      <article key={`${item.source}-${idx}`} className="small-card">
                        <p className="mono">{item.source} | score {item.score}</p>
                        <p className="knowledge-text">
                          {expandedKnowledge[idx] ? item.chunk : shortenText(item.chunk, 180)}
                        </p>
                        {String(item.chunk || '').length > 180 && (
                          <button
                            type="button"
                            className="text-btn knowledge-toggle"
                            onClick={() => toggleKnowledgeChunk(idx)}
                          >
                            {expandedKnowledge[idx] ? 'Show less' : 'Show more'}
                          </button>
                        )}
                      </article>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          )}

          {result?.doctor_mode_enabled && (
            <section className="card">
              <div className="section-head">
                <h2>Doctor Review Mode</h2>
                <p>Edit the generated plan, add notes, and confirm the case.</p>
              </div>

              <form onSubmit={handleReviewSubmit} className="review-grid">
                <input
                  className="input"
                  value={reviewDraft.diagnosis}
                  onChange={(event) => setReviewDraft((prev) => ({ ...prev, diagnosis: event.target.value }))}
                  placeholder="Final diagnosis"
                />

                <input
                  className="input"
                  value={asCsv(reviewDraft.differential_diagnosis)}
                  onChange={(event) => setReviewDraft((prev) => ({ ...prev, differential_diagnosis: event.target.value }))}
                  placeholder="Differentials (comma separated)"
                />

                <input
                  className="input"
                  value={reviewDraft.risk_level}
                  onChange={(event) => setReviewDraft((prev) => ({ ...prev, risk_level: event.target.value }))}
                  placeholder="Risk level"
                />

                <input
                  className="input"
                  value={asCsv(reviewDraft.suggested_tests)}
                  onChange={(event) => setReviewDraft((prev) => ({ ...prev, suggested_tests: event.target.value }))}
                  placeholder="Suggested tests (comma separated)"
                />

                <input
                  className="input"
                  value={asCsv(reviewDraft.treatment_plan)}
                  onChange={(event) => setReviewDraft((prev) => ({ ...prev, treatment_plan: event.target.value }))}
                  placeholder="Treatment plan (comma separated)"
                />

                <textarea
                  rows={3}
                  className="input"
                  value={reviewDraft.referral}
                  onChange={(event) => setReviewDraft((prev) => ({ ...prev, referral: event.target.value }))}
                  placeholder="Referral recommendation"
                />

                <input
                  className="input"
                  value={reviewDraft.confidence_score}
                  onChange={(event) => setReviewDraft((prev) => ({ ...prev, confidence_score: event.target.value }))}
                  placeholder="Confidence score"
                />

                <textarea
                  rows={3}
                  className="input"
                  value={reviewDraft.notes}
                  onChange={(event) => setReviewDraft((prev) => ({ ...prev, notes: event.target.value }))}
                  placeholder="Doctor notes"
                />

                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={reviewDraft.confirmed}
                    onChange={(event) => setReviewDraft((prev) => ({ ...prev, confirmed: event.target.checked }))}
                  />
                  <span>Confirm case</span>
                </label>

                <button type="submit" className="btn btn-primary full">Save Doctor Review</button>
              </form>
            </section>
          )}
        </div>
      </main>

      {expandedImage && (
        <div className="image-modal-backdrop" onClick={() => setExpandedImage(null)}>
          <div className="image-modal" onClick={(event) => event.stopPropagation()}>
            <div className="image-modal-header">
              <p className="strong">{expandedImage.title}</p>
              <button type="button" className="text-btn" onClick={() => setExpandedImage(null)}>
                Close
              </button>
            </div>
            <img src={expandedImage.src} alt={expandedImage.title} className="image-modal-image" />
          </div>
        </div>
      )}
    </div>
  )
}

function MetricList({ title, items = [], icon = null }) {
  return (
    <div className="metric-block">
      <p className="tiny-label inline-row">
        {icon}
        {title}
      </p>
      {items.length === 0 && <p className="muted">No items available.</p>}
      <ul className="metric-list">
        {items.map((item, idx) => (
          <li key={`${item}-${idx}`} className="metric-item">
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}

function ToothLogo() {
  return (
    <svg
      className="tooth-logo"
      viewBox="0 0 64 64"
      role="img"
      aria-label="Orisight logo"
    >
      <defs>
        <clipPath id="orisightToothClip">
          <path d="M32 6c-8.8 0-14.5 6.3-14.5 15.9 0 2.5.5 5.2 1.3 7.8 1.8 5.8 2.5 10.5 2.8 15.6.1 2.1 1.4 3.7 3.3 3.9 2 .2 3.8-.9 4.6-2.7l2.5-5.5 2.5 5.5c.8 1.8 2.6 2.9 4.6 2.7 1.9-.2 3.2-1.8 3.3-3.9.3-5.1 1-9.8 2.8-15.6.8-2.6 1.3-5.3 1.3-7.8C46.5 12.3 40.8 6 32 6Z" />
        </clipPath>
      </defs>

      <g clipPath="url(#orisightToothClip)">
        <rect x="0" y="0" width="32" height="32" fill="#4285F4" />
        <rect x="32" y="0" width="32" height="32" fill="#EA4335" />
        <rect x="0" y="32" width="32" height="32" fill="#34A853" />
        <rect x="32" y="32" width="32" height="32" fill="#FBBC05" />
      </g>

      <path
        d="M32 6c-8.8 0-14.5 6.3-14.5 15.9 0 2.5.5 5.2 1.3 7.8 1.8 5.8 2.5 10.5 2.8 15.6.1 2.1 1.4 3.7 3.3 3.9 2 .2 3.8-.9 4.6-2.7l2.5-5.5 2.5 5.5c.8 1.8 2.6 2.9 4.6 2.7 1.9-.2 3.2-1.8 3.3-3.9.3-5.1 1-9.8 2.8-15.6.8-2.6 1.3-5.3 1.3-7.8C46.5 12.3 40.8 6 32 6Z"
        fill="none"
        stroke="#fff"
        strokeWidth="2.6"
      />
      <path
        d="M23.5 21.6c1.5-2 4-3.1 8.5-3.1s7 1.1 8.5 3.1M32 18.6v10.8"
        fill="none"
        stroke="#fff"
        strokeWidth="1.9"
        strokeLinecap="round"
      />
    </svg>
  )
}

function shortenText(value, maxChars) {
  const text = String(value || '').trim()
  if (text.length <= maxChars) return text
  return `${text.slice(0, maxChars).trimEnd()}...`
}

function toList(value) {
  if (Array.isArray(value)) return value.map((x) => String(x).trim()).filter(Boolean)
  return String(value || '')
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean)
}

function asCsv(value) {
  if (Array.isArray(value)) return value.join(', ')
  return value || ''
}

function formatFileSize(bytes) {
  if (!bytes || Number.isNaN(bytes)) return '0 KB'
  const kb = bytes / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  return `${(kb / 1024).toFixed(1)} MB`
}
