import React, { useEffect, useRef, useState } from 'react'
import { Activity, ArrowDownRight, ArrowRight, Bell, Bot, Check, ChevronDown, ClipboardCheck, Database, FileText, LayoutDashboard, LockKeyhole, Mail, Moon, Send, ShieldAlert, Sun, TrendingUp, X } from 'lucide-react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const navigation = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'analyst', label: 'AI Analyst', icon: Bot },
  { id: 'actions', label: 'Actions & Approvals', icon: ClipboardCheck },
]
const api = async (path, options = {}) => {
  const accessToken = window.sessionStorage.getItem('argos-access-token')
  const response = await fetch(`/api${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}), ...(options.headers || {}) } })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`)
  return body
}
const money = value => value == null || !Number.isFinite(Number(value)) ? '—' : new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(Number(value))
const number = value => value == null || !Number.isFinite(Number(value)) ? '—' : new Intl.NumberFormat('en-IN', { maximumFractionDigits: 1 }).format(Number(value))
const list = value => Array.isArray(value) ? value : []
const label = value => String(value || '').replaceAll('_', ' ').replace(/\b\w/g, x => x.toUpperCase())
const statuses = ['PENDING', 'APPROVED', 'REJECTED', 'EXECUTED']
function routeQuestion(question) {
  const q = question.toLowerCase()
  const asksProfitAction = /\bprofit(?:s|ability)?\b|\bmargin\b/.test(q) && /\b(improv\w*|increas\w*|gain\w*|grow\w*|boost\w*|maximi[sz]\w*|rais\w*|recommend\w*|should|how can|how to|what should)\b/.test(q)
  const policy = /\b(policy|policies|procedure|procedures|compliance|compliant|control|controls|approval rule)\b/.test(q)
  const specialist = asksProfitAction || /\b(why|root cause|driver|drivers|contribut|changed|change|increase|increased|improve|improved|decreased|declined|forecast|forecasting|predict|prediction|future|outlook|recommend|recommendation|suggest|should we|should i|next step|coordinate|investigate)\b/.test(q)
  if (policy && !specialist) return '/policy/ask'
  if (specialist) return '/agents/query'
  return '/analyst/ask'
}

function Card({ children, className = '' }) { return <section className={`panel ${className}`}>{children}</section> }
function Metric({ title, value, detail, icon: Icon, tone = '' }) { return <Card className="argos-metric"><div className="argos-metric-head"><span>{title}</span><span className={`argos-metric-icon ${tone}`}><Icon size={17} /></span></div><strong>{value}</strong><small>{detail}</small></Card> }
function Status({ value }) { const text = String(value || 'unknown').toLowerCase(); return <span className={`badge badge-${text}`}>{label(text)}</span> }
function Table({ rows, columns, empty = 'Nothing to show yet.' }) {
  return <div className="argos-table-wrap"><table className="argos-table"><thead><tr>{columns.map(column => <th key={column.key}>{column.label}</th>)}</tr></thead><tbody>{rows.length ? rows.map((row, i) => <tr key={row.action_id || row.id || i}>{columns.map(column => <td key={column.key}>{column.render ? column.render(row[column.key], row) : String(row[column.key] ?? '—')}</td>)}</tr>) : <tr><td className="argos-empty" colSpan={columns.length}>{empty}</td></tr>}</tbody></table></div>
}
function PanelTitle({ title, detail, action }) { return <div className="argos-panel-title"><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action}</div> }
export function formatStructuredAnswer(response) {
  const analysis = response?.analysis || response || {}
  const sections = [['executive_summary', 'Executive summary'], ['performance_analysis', 'Performance'], ['root_cause_analysis', 'Root-cause findings'], ['forecast_analysis', 'Forecast'], ['trends', 'Trends'], ['risks', 'Risks'], ['opportunities', 'Opportunities'], ['attention_items', 'Management attention'], ['recommendations', 'Recommended next steps'], ['data_limitations', 'Data limitations']]
  return sections.map(([key, title]) => {
    const texts = list(analysis[key]).map(item => typeof item?.text === 'string' ? item.text.trim() : '').filter(Boolean)
    return texts.length ? `${title}\n${texts.map(text => `• ${text}`).join('\n')}` : ''
  }).filter(Boolean).join('\n\n')
}

export default function App() {
  const [page, setPage] = useState('overview')
  const [dark, setDark] = useState(localStorage.getItem('argos-theme') === 'dark' || (!localStorage.getItem('argos-theme') && localStorage.getItem('finop-theme') === 'dark'))
  const [overview, setOverview] = useState(null)
  const [actions, setActions] = useState([])
  const [audit, setAudit] = useState([])
  const [loading, setLoading] = useState(false)
  const [connected, setConnected] = useState(true)
  const [error, setError] = useState('')
  const [toast, setToast] = useState('')
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)
  const [employee, setEmployee] = useState(() => window.sessionStorage.getItem('argos-employee') || '')
  const [signedIn, setSignedIn] = useState(() => window.sessionStorage.getItem('argos-signed-in') === 'true')
  const [loginError, setLoginError] = useState('')
  const [signingIn, setSigningIn] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('argos-theme', dark ? 'dark' : 'light')
  }, [dark])

  const refreshOverview = async () => {
    const [dashboard, workflow, history] = await Promise.all([api('/dashboard'), api('/actions'), api('/audit')])
    setOverview(dashboard)
    setActions(list(workflow.actions))
    setAudit(list(history.events).slice().reverse())
    setConnected(true)
  }
  const refreshActions = async () => {
    const [workflow, history] = await Promise.all([api('/actions'), api('/audit')])
    setActions(list(workflow.actions))
    setAudit(list(history.events).slice().reverse())
    setConnected(true)
  }

  useEffect(() => {
    if (!signedIn) return undefined
    let active = true
    setLoading(true)
    setError('')
    const load = page === 'overview' ? refreshOverview() : page === 'actions' ? refreshActions() : Promise.resolve()
    Promise.resolve(load).catch(e => { if (active) { setError(e.message); setConnected(false) } }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [page, signedIn])

  async function signIn(event) {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    const email = String(form.get('employeeEmail') || '').trim()
    const token = String(form.get('accessToken') || '').trim()
    if (!email) { setLoginError('Enter your work email to continue.'); return }
    setSigningIn(true)
    setLoginError('')
    if (token) window.sessionStorage.setItem('argos-access-token', token)
    else window.sessionStorage.removeItem('argos-access-token')
    try {
      await api('/dashboard')
      window.sessionStorage.setItem('argos-employee', email)
      window.sessionStorage.setItem('argos-signed-in', 'true')
      setEmployee(email)
      setSignedIn(true)
    } catch (e) {
      window.sessionStorage.removeItem('argos-access-token')
      setLoginError(e.message.includes('Authentication required') ? 'Access was not accepted. Check the access token with your Argos administrator.' : `Could not connect to Argos. ${e.message}`)
    } finally { setSigningIn(false) }
  }

  function signOut() {
    window.sessionStorage.removeItem('argos-access-token')
    window.sessionStorage.removeItem('argos-employee')
    window.sessionStorage.removeItem('argos-signed-in')
    setEmployee('')
    setSignedIn(false)
  }

  const notify = text => { setToast(text); window.setTimeout(() => setToast(''), 3200) }
  async function ask(text = question) {
    const prompt = text.trim()
    if (prompt.length < 3 || busy) return
    setQuestion('')
    setMessages(current => [...current, { role: 'user', text: prompt }])
    setBusy(true)
    setError('')
    try {
      const result = await api(routeQuestion(prompt), { method: 'POST', body: JSON.stringify({ question: prompt }) })
      setMessages(current => [...current, { role: 'assistant', result }])
    } catch (e) {
      const message = e.message.includes('unavailable') ? e.message : `I couldn’t complete that analysis. ${e.message}`
      setMessages(current => [...current, { role: 'error', text: message }])
    } finally { setBusy(false) }
  }

  async function decision(actionId, operation, body) {
    try {
      await api(`/actions/${actionId}/${operation}`, { method: 'POST', body: JSON.stringify(body) })
      await refreshActions()
      if (page === 'overview') await refreshOverview()
      notify(operation === 'execute' ? 'Action executed and recorded.' : operation === 'approve' ? 'Action approved.' : 'Action rejected.')
    } catch (e) { setError(e.message) }
  }
  async function proposeActions() {
    try {
      const result = await api('/actions/propose', { method: 'POST', body: JSON.stringify({}) })
      await refreshActions()
      notify(`${result.created_action_ids?.length || 0} action proposal(s) created.`)
      return result
    } catch (e) { setError(e.message); return null }
  }
  async function generateRecommendation() {
    try {
      const result = await api('/agents/query', { method: 'POST', body: JSON.stringify({ question: 'Based on the latest financial performance, associated changes, available forecasts, and relevant policies, what should we do next?' }) })
      const recommendations = list(result.analysis?.recommendations)
      setOverview(current => current ? { ...current, recommendations } : current)
      notify(recommendations.length ? 'Evidence-grounded recommendation is ready.' : 'No supported recommendation was returned. Review the analyst evidence.')
    } catch (e) { setError(e.message) }
  }
  async function createOverviewProposals() {
    const result = await proposeActions()
    if (result) setPage('actions')
  }
  async function refreshCurrent() {
    setLoading(true)
    setError('')
    try {
      if (page === 'overview') await refreshOverview()
      else if (page === 'actions') await refreshActions()
      else { await api('/health'); setConnected(true) }
    } catch (e) { setError(e.message); setConnected(false) }
    finally { setLoading(false) }
  }
  const selected = navigation.find(item => item.id === page) || navigation[0]

  if (!signedIn) return <LoginPage dark={dark} setDark={setDark} onSubmit={signIn} error={loginError} busy={signingIn} />

  return <div className={`argos-shell ${dark ? 'argos-dark' : 'argos-light'}`}>
    <header className="argos-site-header"><div className="argos-header-inner">
      <a className="argos-brand-lockup" href="#overview" onClick={event => { event.preventDefault(); setPage('overview') }}><span className="argos-brand-mark"><i /></span><span><b>Argos</b><small>Agentic financial operations</small></span></a>
      <button className="argos-workspace" aria-label="Selected workspace and dataset"><span className="workspace-avatar">A</span><span><b>Finance workspace</b><small>Amazon Sales & Trading Insights</small></span><ChevronDown size={14} /></button>
      <nav className="argos-nav" aria-label="Main navigation">{navigation.map(({ id, label: title, icon: Icon }) => <button key={id} className={`argos-nav-link ${page === id ? 'active' : ''}`} onClick={() => setPage(id)}><Icon size={15} /><span>{title}</span>{id === 'actions' && actions.filter(action => action.status === 'PENDING').length > 0 && <span className="argos-nav-count">{actions.filter(action => action.status === 'PENDING').length}</span>}</button>)}</nav>
      <div className="argos-header-actions"><span className={`argos-connection ${connected ? '' : 'offline'}`}><i />{connected ? 'Reports connected' : 'Reports unavailable'}</span><button className="argos-header-icon" aria-label="Open pending actions" onClick={() => setPage('actions')}><Bell size={17} />{actions.some(action => action.status === 'PENDING') && <i />}</button><button className="argos-refresh" onClick={refreshCurrent} disabled={loading}><Activity size={15} /><span>{loading ? 'Refreshing' : 'Refresh'}</span></button><button className="argos-theme-toggle" onClick={() => setDark(value => !value)} aria-label={`Switch to ${dark ? 'light' : 'dark'} mode`}><span className="argos-theme-switch">{dark ? <Sun size={12} /> : <Moon size={12} />}</span><span>{dark ? 'Light mode' : 'Dark mode'}</span></button><button className="argos-profile argos-profile-button" onClick={signOut} title={`Sign out ${employee}`}>{employee.split('@')[0].slice(0, 2).toUpperCase() || 'AN'}</button></div>
    </div></header>
    <main className="argos-main">
      <div className="argos-content">
        {page !== 'overview' && <div className="argos-page-heading"><div><div className="eyebrow">ARGOS WORKSPACE</div><h1>{selected.label}</h1><p>{page === 'analyst' ? 'Ask a question and get a grounded answer from your financial reports and policies.' : 'Review recommendations, approvals, and recorded workflow activity.'}</p></div></div>}
        {error && <div className="global-error"><ShieldAlert size={16} />{error}<button onClick={() => setError('')} aria-label="Dismiss"><X size={15} /></button></div>}
        {loading && !overview && page === 'overview' ? <div className="state"><span className="spinner" />Loading your financial workspace…</div> : null}
        {page === 'overview' && overview && <Overview data={overview} actions={actions} audit={audit} onNavigate={setPage} onApprove={(id) => decision(id, 'approve', { actor: 'Dashboard operator', reason: 'Reviewed and approved in Argos.' })} onRecommend={generateRecommendation} onPropose={createOverviewProposals} />}
        {page === 'analyst' && <Analyst messages={messages} question={question} setQuestion={setQuestion} busy={busy} onAsk={ask} />}
        {page === 'actions' && <ActionsWorkspace actions={actions} audit={audit} onDecision={decision} onPropose={proposeActions} />}
        <footer className="footer argos-footer"><span>Argos · Financial Operations</span><span>Connected to your finance reports</span></footer>
      </div>
    </main>
    {toast && <div role="status" className="toast">{toast}</div>}
  </div>
}

function LoginPage({ dark, setDark, onSubmit, error, busy }) {
  return <div className={`argos-shell ${dark ? 'argos-dark' : 'argos-light'}`}>
    <div className="argos-login-top"><a className="argos-brand-lockup" href="#login"><span className="argos-brand-mark"><i /></span><span><b>Argos</b><small>Agentic financial operations</small></span></a><button className="argos-theme-toggle" onClick={() => setDark(value => !value)} aria-label={`Switch to ${dark ? 'light' : 'dark'} mode`}><span className="argos-theme-switch">{dark ? <Sun size={13} /> : <Moon size={13} />}</span><span>{dark ? 'Light mode' : 'Dark mode'}</span></button></div>
    <main className="argos-login-main"><section className="argos-login-story"><span className="argos-hero-kicker"><i /> FINANCIAL OPERATIONS WORKSPACE</span><h1>See the full story<br />behind your numbers.</h1><p>Bring reports, investigations, forecasts, and recommended actions together in one clear workspace.</p><div className="argos-login-points"><div><span>01</span><p><b>Understand performance</b><small>Track revenue, orders, and financial risk.</small></p></div><div><span>02</span><p><b>Ask Argos</b><small>Investigate results and find supporting evidence.</small></p></div><div><span>03</span><p><b>Review next steps</b><small>Manage proposals with human approval.</small></p></div></div></section>
      <form className="argos-login-card" onSubmit={onSubmit}><span className="argos-login-icon"><LockKeyhole size={19} /></span><h2>Employee sign in</h2><p>Sign in to open your finance workspace.</p><label htmlFor="employeeEmail">Work email</label><div className="argos-login-input"><Mail size={16} /><input id="employeeEmail" name="employeeEmail" type="email" autoComplete="username" placeholder="name@company.com" required /></div><label htmlFor="accessToken">Workspace access token <span>if required</span></label><div className="argos-login-input"><LockKeyhole size={16} /><input id="accessToken" name="accessToken" type="password" autoComplete="current-password" placeholder="Enter your access token" /></div>{error && <div className="argos-login-error" role="alert"><ShieldAlert size={15} />{error}</div>}<button className="argos-login-submit" type="submit" disabled={busy}>{busy ? 'Connecting…' : 'Continue to Argos'}<ArrowRight size={16} /></button><div className="argos-login-note"><ShieldAlert size={14} /><span>Use the workspace token provided by your administrator when access control is enabled. In local development, email is only used as a display name.</span></div></form>
    </main><footer className="argos-login-footer"><span>Argos · Financial Operations</span><span>Secure workspace access</span></footer>
  </div>
}

export function Overview({ data, actions, audit, onNavigate, onApprove, onRecommend, onPropose }) {
  const summary = data.financial_summary || {}, kpis = data.kpis || {}, risk = data.risk || {}
  const revenue = summary.total_revenue ?? summary.revenue ?? kpis.total_revenue
  const orders = summary.total_orders ?? kpis.total_orders
  const quantity = summary.total_quantity ?? kpis.total_quantity
  const trend = list(data.monthly_trends).slice(-18).map(item => ({ ...item, month: item.month || item._month || item.period || item.date, revenue: Number(item.revenue || 0) }))
  const forecasts = list(data.forecast_summary)
  const pending = actions.filter(action => action.status === 'PENDING')
  const activity = audit.slice(0, 5)
  const attention = list(data.attention_items).slice(0, 4)
  const recommendations = list(data.recommendations)
  const latestChange = list(data.latest_root_cause)[0]
  const contributors = latestChange?.contributors && typeof latestChange.contributors === 'object'
    ? Object.values(latestChange.contributors).flatMap(items => Array.isArray(items) ? items : []).slice(0, 2).map(item => [item?.dimension, item?.value || item?.name].filter(Boolean).join(': ')).filter(Boolean).join(', ')
    : ''
  const outlook = forecasts[0]
  const outlookDirection = typeof outlook?.trend === 'object' ? outlook.trend?.direction : outlook?.trend || outlook?.forecast_trend
  const trail = [
    { label: 'WHAT HAPPENED', text: latestChange ? `${label(latestChange.metric)} changed ${number(latestChange.percentage_change)}% in ${latestChange.period || 'the latest reported period'}.` : list(risk.risk_factors)[0] || 'No investigation finding is available in the current reports.' },
    { label: 'WHY IT MAY HAVE HAPPENED', text: contributors || latestChange?.evidence?.[0] || (latestChange ? 'Contributor details are included in the investigation evidence.' : 'No associated contributors are available without an investigation finding.') },
    { label: 'WHAT WILL HAPPEN NEXT?', text: outlook ? `${label(outlook.metric)} outlook: ${label(outlookDirection || outlook.status || 'available')}.` : 'No forecast summary is available in the current reports.' },
    { label: 'WHAT SHOULD WE DO?', text: recommendations[0]?.text || 'Generate an evidence-grounded recommendation from the available reports.' },
  ]
  return <>
    <section className="argos-hero"><div className="argos-hero-copy"><span className="argos-hero-kicker"><i /> AGENTIC FINANCIAL OPERATIONS</span><h1>Your ledger,<br />investigated — not just reported.</h1><p>Argos connects financial performance, investigation, forecasts, and next steps in one clear workspace.</p><div className="argos-hero-actions"><button className="argos-hero-primary" onClick={() => onNavigate('analyst')}>Ask Argos <ArrowRight size={16} /></button><button className="argos-hero-secondary" onClick={() => onNavigate('actions')}>Review approvals <span>{pending.length}</span></button></div></div><Card className="argos-trail-card"><div className="argos-trail-heading"><span className="argos-trail-mark"><Activity size={16} /></span><div><b>Latest financial trail</b><small>What happened · why · outlook · recommended action</small></div></div><div className="argos-trail-steps">{trail.map((step, index) => <div className="argos-trail-step" key={step.label}><span className="argos-trail-number">{index + 1}</span><div><b>{step.label}</b><p>{step.text}</p>{index === 1 && latestChange && <small>Associated with the change; not proof of cause.</small>}{index === 3 && <div className="argos-trail-actions">{recommendations.length ? <button onClick={onPropose}>Create action proposals</button> : <button onClick={onRecommend}>Generate recommendation</button>}</div>}</div></div>)}</div></Card></section>
    <div className="metric-grid argos-kpis"><Metric title="Revenue" value={money(revenue)} detail="Total recorded revenue" icon={Activity} tone="purple" /><Metric title="Orders" value={number(orders)} detail="Unique orders" icon={ClipboardCheck} tone="blue" /><Metric title="Quantity" value={number(quantity)} detail="Units recorded" icon={Database} tone="green" /><Metric title="Risk" value={risk.risk_level ? label(risk.risk_level) : number(risk.risk_score)} detail={risk.risk_score != null ? `${number(risk.risk_score)} / 100 · Current risk score` : 'Risk level from reports'} icon={ShieldAlert} tone="amber" /></div>
    <div className="argos-overview-grid">
      <Card className="argos-chart-card"><PanelTitle title="Revenue trend" detail="Monthly revenue from your reports" /><div className="chart-box argos-chart">{trend.length ? <ResponsiveContainer width="100%" height="100%"><AreaChart data={trend} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}><defs><linearGradient id="argosRevenue" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#5266df" stopOpacity={.2} /><stop offset="95%" stopColor="#5266df" stopOpacity={0} /></linearGradient></defs><CartesianGrid vertical={false} strokeDasharray="3 5" /><XAxis dataKey="month" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis tickFormatter={number} tick={{ fontSize: 9 }} axisLine={false} tickLine={false} /><Tooltip formatter={value => money(value)} /><Area type="monotone" dataKey="revenue" name="Revenue" stroke="#5266df" strokeWidth={2.5} fill="url(#argosRevenue)" /></AreaChart></ResponsiveContainer> : <div className="chart-empty">Monthly revenue history is not available in this report.</div>}</div></Card>
      <Card className="argos-side-card"><PanelTitle title="Forecast snapshot" detail="Expected direction across available outlooks" /><div className="argos-forecast-list">{forecasts.slice(0, 3).map((forecast, index) => <div className="argos-forecast-row" key={`${forecast.metric}-${index}`}><span>{label(forecast.metric)}</span><b>{label(forecast.trend || forecast.forecast_trend || forecast.status || 'Available')}</b><small>{forecast.selected_model?.name || forecast.forecast_period || 'Forecast summary'}</small></div>)}{!forecasts.length && <div className="empty-note">No forecast summary is available yet.</div>}</div></Card>
    </div>
    <div className="argos-lower-grid">
      <Card><PanelTitle title="Risk & attention" detail="Items surfaced by your financial reports" /><div className="argos-attention-list">{attention.map((item, index) => <div className="argos-attention-item" key={index}><span><ShieldAlert size={15} /></span><p>{item.text || String(item)}</p></div>)}{!attention.length && list(data.latest_root_cause).slice(0, 3).map((item, index) => <div className="argos-attention-item" key={index}><span><ArrowDownRight size={15} /></span><p><b>{label(item.metric)} · {item.period}</b><br />{item.evidence?.[0] || `${number(item.percentage_change)}% period change`}</p></div>)}{!attention.length && !list(data.latest_root_cause).length && <div className="empty-note">No items need attention right now.</div>}</div><button className="text-action" onClick={() => onNavigate('analyst')}>Investigate with Argos <ArrowRight size={15} /></button></Card>
      <Card><PanelTitle title="Pending approvals" detail={`${pending.length} action(s) awaiting review`} action={<button className="text-action" onClick={() => onNavigate('actions')}>View all <ArrowRight size={14} /></button>} />{pending.length ? <div className="argos-pending-list">{pending.slice(0, 4).map(action => <div className="argos-pending-row" key={action.action_id}><div><b>{action.description || label(action.type)}</b><small>{label(action.type)} · {label(action.priority || 'normal')}</small></div><button onClick={() => onApprove(action.action_id)}>Approve</button></div>)}</div> : <div className="empty-note">No actions are waiting for approval.</div>}</Card>
    </div>
    <Card className="argos-activity-card"><PanelTitle title="Recent activity" detail="Latest recorded workflow events" action={<button className="text-action" onClick={() => onNavigate('actions')}>Open actions <ArrowRight size={14} /></button>} /><div className="argos-activity-list">{activity.slice(0, 4).map((event, index) => <div className="argos-activity-row" key={event.id || event.timestamp || index}><span className="argos-activity-icon"><Activity size={15} /></span><div><b>{label(event.event || event.action || 'Workflow update')}</b><small>{event.reason || event.action_id || event.actor || 'Recorded in the audit history'}</small></div><time>{event.timestamp ? new Date(event.timestamp).toLocaleString() : ''}</time></div>)}{!activity.length && <div className="empty-note">Workflow activity will appear here.</div>}</div></Card>
  </>
}

// Retained as a compatibility export for existing dashboard integrations.
export function Dashboard({ data = {}, money: formatMoney = money, number: formatNumber = number }) {
  const summary = data.financial_summary || {}, kpis = data.kpis || {}, risk = data.risk || {}
  return <div className="metric-grid argos-kpis">
    <Metric title="Revenue" value={formatMoney(summary.total_revenue ?? summary.revenue ?? kpis.total_revenue)} detail="Total recorded revenue" icon={Activity} />
    <Metric title="Orders" value={formatNumber(summary.total_orders ?? kpis.total_orders)} detail="Unique orders" icon={ClipboardCheck} />
    <Metric title="Quantity" value={formatNumber(summary.total_quantity ?? kpis.total_quantity)} detail="Units recorded" icon={Database} />
    <Metric title="Risk" value={risk.risk_level ? label(risk.risk_level) : formatNumber(risk.risk_score)} detail={risk.risk_score != null ? `${formatNumber(risk.risk_score)} / 100 · Current risk score` : 'Risk level from reports'} icon={ShieldAlert} />
  </div>
}

// Retained for clients that used the former single-answer analyst component.
export function Assistant({ title, description, question, setQuestion, answer, busy, onSend }) {
  const answerText = answer?.answer || formatStructuredAnswer(answer)
  return <Card className="argos-chat-card"><div className="argos-chat-heading"><span className="argos-analyst-mark"><Bot size={20} /></span><div><h2>{title}</h2><p>{description}</p></div></div><div className="argos-conversation">{answer ? <article className="argos-answer"><div className="argos-answer-label"><b>Argos</b><Status value={answer.status || 'answered'} /></div>{answerText && <p className="argos-answer-summary">{answerText}</p>}</article> : <div className="argos-chat-welcome"><h3>Ask a question about your financial operations</h3></div>}</div><form className="argos-composer" onSubmit={event => { event.preventDefault(); onSend?.() }}><textarea value={question || ''} onChange={event => setQuestion?.(event.target.value)} placeholder="Ask a question…" rows={2} /><div><span /><button className="primary-button" disabled={busy || (question || '').trim().length < 3}><Send size={16} /> Ask Argos</button></div></form></Card>
}

function Analyst({ messages, question, setQuestion, busy, onAsk }) {
  const suggestions = ['Summarize the latest revenue trend', 'Why did revenue change?', 'What does the forecast show?', 'What policy applies to this decision?']
  const endRef = useRef(null)
  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }) }, [messages, busy])
  return <div className="argos-analyst-layout"><div className="argos-chat-card panel"><div className="argos-chat-heading"><span className="argos-analyst-mark"><Bot size={20} /></span><div><h2>Your financial analyst</h2><p>Ask a question. Argos brings together the relevant financial, forecast, policy, and recommendation evidence.</p></div><span className="argos-online"><i /> Ready</span></div>
    <div className="argos-conversation">{!messages.length && !busy && <div className="argos-chat-welcome"><div className="argos-chat-orb"><Bot size={27} /></div><h3>What would you like to understand?</h3><p>Ask about performance, changes, forecasts, risks, or policy.</p><div className="argos-suggestions">{suggestions.map(text => <button key={text} onClick={() => onAsk(text)}>{text}<ArrowRight size={14} /></button>)}</div></div>}
      {messages.map((message, index) => message.role === 'user' ? <div className="argos-user-message" key={index}><span>You</span><p>{message.text}</p></div> : message.role === 'error' ? <div className="argos-error-message" key={index}><ShieldAlert size={17} /><p>{message.text}</p></div> : <AnalystAnswer key={index} result={message.result} />)}
      {busy && <div className="argos-thinking"><span className="spinner" /><div><b>Looking into that…</b><small>Gathering the relevant report and policy evidence.</small></div></div>}<div ref={endRef} />
    </div>
    <form className="argos-composer" onSubmit={event => { event.preventDefault(); onAsk() }}><textarea value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); onAsk() } }} placeholder="Ask anything about your financial operations…" rows={2} /><div><small>Answers are grounded in your reports and indexed policies.</small><button className="primary-button" disabled={busy || question.trim().length < 3}><Send size={16} /> Ask Argos</button></div></form>
  </div></div>
}
function AnalystAnswer({ result }) {
  const groups = result?.analysis || {}
  const isAgentResult = Array.isArray(groups.facts) || Array.isArray(groups.findings) || Array.isArray(groups.forecasts) || Array.isArray(groups.policies)
  const isPolicyResult = Array.isArray(result?.claims)
  const isProfitAction = (/\bprofit(?:s|ability)?\b|\bmargin\b/i.test(result?.question || '')) && /\b(improv\w*|increas\w*|gain\w*|grow\w*|boost\w*|maximi[sz]\w*|rais\w*|recommend\w*|should|how can|how to|what should)\b/i.test(result?.question || '')
  const agentOrder = isProfitAction
    ? [['facts', 'What happened?'], ['findings', 'Why it may have happened'], ['forecasts', 'What may happen next'], ['policies', 'Relevant policy']]
    : [['facts', 'Financial evidence'], ['findings', 'What changed'], ['forecasts', 'Forecast'], ['policies', 'Policy guidance']]
  const directOrder = [['performance_analysis', 'Performance'], ['executive_summary', 'Financial evidence'], ['root_cause_analysis', 'Change analysis'], ['forecast_analysis', 'Forecast'], ['trends', 'Trend'], ['risks', 'Risk'], ['attention_items', 'Management attention'], ['data_limitations', 'Data limitation']]
  const content = isAgentResult
    ? agentOrder.flatMap(([key, title]) => list(groups[key]).map(item => ({ title, item, text: item?.text || String(item) })))
    : isPolicyResult
      ? list(result.claims).map(item => ({ title: 'Policy guidance', item, text: item?.text || String(item) }))
      : directOrder.flatMap(([key, title]) => list(groups[key]).map(item => ({ title, item, text: item?.text || String(item) })))
  const recommendations = list(groups.recommendations)
  const summary = isAgentResult
    ? (isProfitAction
        ? (list(groups.recommendations)[0]?.text || 'I’ll connect the available sales evidence, related changes, forecast, and policies to identify practical next steps. Net profit impact cannot be quantified without product cost data.')
        : (['findings', 'forecasts', 'policies', 'recommendations', 'facts'].flatMap(key => list(groups[key]))[0]?.text || content[0]?.text))
    : isPolicyResult
      ? result.answer || content[0]?.text
      : list(groups.performance_analysis)[0]?.text || list(groups.executive_summary)[0]?.text || content[0]?.text || result.answer
  const sources = list(result?.source_citations).length ? list(result.source_citations) : list(result?.sources).length ? list(result.sources) : list(result?.retrieved_documents).map(document => ({ source: document.metadata?.source, page: document.metadata?.page, section: document.metadata?.section }))
  const capabilityNames = { financial_analyst: 'Financial reports', root_cause_agent: 'Change analysis', forecast_agent: 'Forecast', policy_rag_agent: 'Policy library', recommendation_agent: 'Recommendations' }
  const selected = list(result?.selected_agents).map(name => capabilityNames[name] || label(name.replace(/_agent$/, '').replaceAll('_', ' ')))
  if (!selected.length && isPolicyResult) selected.push('Policy library')
  if (!selected.length && !isAgentResult && !isPolicyResult) selected.push('Financial reports')
  const remaining = content.filter(entry => entry.text !== summary)
  return <article className="argos-answer"><div className="argos-answer-label"><span className="argos-analyst-mark small"><Bot size={15} /></span><b>Argos</b><Status value={result?.status || 'complete'} /></div><p className="argos-answer-summary">{summary}</p>
    {remaining.length > 0 && <div className="argos-answer-points">{remaining.slice(0, 6).map(({ title, item, text }, index) => <div key={`${title}-${index}`}><span>{title}</span><p>{text}</p>{list(item?.evidence_refs || item?.evidence_ids || item?.source_chunk_ids).length > 0 && <small>Evidence attached</small>}</div>)}</div>}
    {recommendations.length > 0 && <section className="argos-recommendations"><h3>{isProfitAction ? 'What should we do?' : 'Recommended next steps'}</h3>{isProfitAction && <p className="argos-limitations">These are evidence-based sales and cost-to-serve actions, not a quantified profit forecast. Add product cost/COGS to measure profit impact.</p>}{recommendations.map((item, index) => <p key={index}><Check size={15} />{item.text || String(item)}</p>)}</section>}
    {sources.length > 0 && <details className="argos-evidence"><summary>Evidence & sources <span>{sources.length}</span></summary><div className="argos-citations">{sources.map((source, index) => { const raw = typeof source === 'string' ? source : source.source || source.title || ''; const file = raw.split(/[\\/]/).pop()?.replace(/\.json$|\.pdf$/i, '') || ''; const known = { phase1_financial_summary: 'Financial summary', phase1_data_quality: 'Data quality', phase1_risk: 'Risk assessment', phase1_concentration: 'Sales concentration', phase1_trends: 'Revenue trends', phase2_changes: 'Change analysis', phase3_forecasts: 'Forecasting', financial_report: 'Financial performance report', root_cause_report: 'Change analysis report', forecast_report: 'Forecast report', rag_report: 'Policy source', llm_financial_analysis: 'Financial analysis', agentic_financial_analysis: 'Coordinated analysis' }; const friendly = known[file.toLowerCase()] || file.replace(/^phase\d+_/i, '').replaceAll('_', ' '); return <div key={`${raw}-${index}`}><FileText size={14} /><span><b>{friendly || `Evidence source ${index + 1}`}</b>{(source?.page || source?.section) && <small>{[source.section, source.page && `Page ${source.page}`].filter(Boolean).join(' · ')}</small>}</span></div> })}</div></details>}
    {selected.length > 0 && <details className="argos-evidence"><summary>Sources & analysis</summary><div className="argos-used-capabilities">{selected.map(name => <span key={name}>{name}</span>)}</div>{list(result?.limitations).length > 0 && <p className="argos-limitations">{list(result.limitations).join(' ')}</p>}</details>}
    {content.length === 0 && recommendations.length === 0 && list(result?.limitations).length > 0 && <p className="argos-limitations">{list(result.limitations).join(' ')}</p>}
  </article>
}

function ActionsWorkspace({ actions, audit, onDecision, onPropose }) {
  const [filter, setFilter] = useState('ALL')
  const [confirm, setConfirm] = useState(null)
  const rows = filter === 'ALL' ? actions : actions.filter(action => action.status === filter)
  const columns = [
    { key: 'description', label: 'Recommendation', render: (value, row) => <div className="argos-action-name"><b>{value || label(row.type)}</b><small>{label(row.type)}</small></div> },
    { key: 'priority', label: 'Priority', render: value => <Status value={value || 'normal'} /> },
    { key: 'status', label: 'Status', render: value => <Status value={value} /> },
    { key: 'created_at', label: 'Created', render: (value, row) => row.timestamps?.created_at || value ? new Date(row.timestamps?.created_at || value).toLocaleDateString() : '—' },
    { key: 'action_id', label: 'Review', render: (value, row) => row.status === 'PENDING' ? <div className="table-actions"><button onClick={() => onDecision(value, 'approve', { actor: 'Dashboard operator', reason: 'Reviewed and approved in Argos.' })}>Approve</button><button className="danger-text" onClick={() => onDecision(value, 'reject', { actor: 'Dashboard operator', reason: 'Rejected in Argos.' })}>Reject</button></div> : row.status === 'APPROVED' ? <button className="table-execute" onClick={() => setConfirm(row)}>Execute</button> : '—' },
  ]
  const auditRows = audit.slice(0, 8).map((event, index) => ({ ...event, id: event.id || index, display_event: label(event.event || event.action || 'Workflow update') }))
  return <>
    <div className="argos-actions-intro"><div><h2>Recommendations become actions here.</h2><p>Review proposals, record decisions, and execute approved actions with a human confirmation.</p></div><button className="primary-button" onClick={onPropose}><ClipboardCheck size={16} /> Create action proposals</button></div>
    <div className="argos-status-tabs">{['ALL', ...statuses].map(status => <button key={status} className={filter === status ? 'selected' : ''} onClick={() => setFilter(status)}>{status === 'ALL' ? 'All actions' : label(status)}<span>{status === 'ALL' ? actions.length : actions.filter(action => action.status === status).length}</span></button>)}</div>
    <Card><PanelTitle title="Action proposals" detail="Email notifications, management reports, tickets, and financial alerts" /><Table rows={rows} columns={columns} empty={filter === 'PENDING' ? 'No actions are waiting for approval.' : 'No proposals in this status.'} /></Card>
    <Card className="argos-audit-card"><PanelTitle title="Audit history" detail="Approvals, rejections, executions, and workflow events" /><Table rows={auditRows} columns={[{ key: 'display_event', label: 'Event' }, { key: 'action_id', label: 'Action' }, { key: 'actor', label: 'By' }, { key: 'reason', label: 'Details' }, { key: 'timestamp', label: 'Time', render: value => value ? new Date(value).toLocaleString() : '—' }]} empty="No workflow events have been recorded." /></Card>
    {confirm && <div className="modal-backdrop" role="presentation"><div className="confirm-modal" role="dialog" aria-modal="true"><button className="modal-close" onClick={() => setConfirm(null)} aria-label="Close"><X size={18} /></button><span className="modal-warning"><ShieldAlert size={23} /></span><h2>Confirm action execution</h2><p>Execute <b>{label(confirm.type)}</b> for “{confirm.description}”? This action has been approved and may send an email or create a ticket.</p><div className="modal-actions"><button className="outline-button" onClick={() => setConfirm(null)}>Cancel</button><button className="danger-button" onClick={async () => { await onDecision(confirm.action_id, 'execute', { actor: 'Dashboard operator', confirm: true }); setConfirm(null) }}>Confirm and execute</button></div></div></div>}
  </>
}
