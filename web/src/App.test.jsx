import React from 'react'
import { describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { Assistant, Dashboard, formatStructuredAnswer, Overview } from './App.jsx'

describe('dashboard report binding', () => {
  it('renders supplied financial report values instead of placeholder metrics', () => {
    render(<Dashboard data={{ financial_summary:{total_revenue:123400}, kpis:{total_orders:27,total_quantity:91}, risk:{risk_score:30,risk_level:'MEDIUM'}, monthly_trends:[{month:'2025-01',revenue:123400}] }} money={x=>`₹${Number(x).toLocaleString('en-IN')}`} number={x=>String(x)} onNavigate={()=>{}} />)
    expect(screen.getByText('₹1,23,400')).toBeInTheDocument()
    expect(screen.getByText('27')).toBeInTheDocument()
    expect(screen.getByText('91')).toBeInTheDocument()
    expect(screen.queryByText('$2,48,650')).not.toBeInTheDocument()
  })

  it('shows the four financial questions and connects recommendations to proposals', () => {
    const createProposals = vi.fn()
    render(<Overview data={{
      latest_root_cause: [{ metric: 'revenue', percentage_change: -12, period: '2025-02', contributors: { product: [{ dimension: 'Product', value: 'A' }] } }],
      forecast_summary: [{ metric: 'revenue', trend: 'increasing' }],
      recommendations: [{ text: 'Review Product A with its owner.' }],
    }} actions={[]} audit={[]} onNavigate={() => {}} onApprove={() => {}} onRecommend={() => {}} onPropose={createProposals} />)
    expect(screen.getByText('WHAT HAPPENED')).toBeInTheDocument()
    expect(screen.getByText('WHY IT MAY HAVE HAPPENED')).toBeInTheDocument()
    expect(screen.getByText('WHAT WILL HAPPEN NEXT?')).toBeInTheDocument()
    expect(screen.getByText('WHAT SHOULD WE DO?')).toBeInTheDocument()
    expect(screen.getByText('Product: A')).toBeInTheDocument()
    expect(screen.getByText('Review Product A with its owner.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Create action proposals' }))
    expect(createProposals).toHaveBeenCalledOnce()
  })

  it('offers to generate a recommendation when one is unavailable', () => {
    const generate = vi.fn()
    render(<Overview data={{}} actions={[]} audit={[]} onNavigate={() => {}} onApprove={() => {}} onRecommend={generate} onPropose={() => {}} />)
    expect(screen.getByText('No investigation finding is available in the current reports.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Generate recommendation' }))
    expect(generate).toHaveBeenCalledOnce()
  })
})

describe('AI analyst answer display data', () => {
  it('reads the nested FastAPI analysis object and renders generated text', () => {
    const answer = formatStructuredAnswer({ analysis: {
      root_cause_analysis: [{ text: 'Revenue decreased with lower category sales.', classification: 'interpretation', evidence_ids: ['phase2_changes'] }],
      recommendations: [{ text: 'Review the contributing categories.', classification: 'recommendation', evidence_ids: ['phase2_changes'] }],
    } })
    expect(answer).toContain('Root-cause findings')
    expect(answer).toContain('Revenue decreased with lower category sales.')
    expect(answer).toContain('Review the contributing categories.')
  })
  it('returns empty text for a valid but empty analysis', () => {
    expect(formatStructuredAnswer({ analysis: { risks: [] } })).toBe('')
  })
  it('renders the backend answer field in the analyst chat', () => {
    render(<Assistant title="AI financial analyst" description="Grounded response" question="" setQuestion={()=>{}} answer={{status:'answered',answer:'Revenue is associated with lower product contributions.',analysis:{root_cause_analysis:[{text:'Evidence backed finding.',classification:'interpretation',evidence_ids:['phase2_changes']}],recommendations:[]},sources:['phase2_changes']}} busy={false} onSend={()=>{}} />)
    expect(screen.getByText('Revenue is associated with lower product contributions.')).toBeInTheDocument()
    expect(screen.getByText('Answered')).toBeInTheDocument()
  })
})
