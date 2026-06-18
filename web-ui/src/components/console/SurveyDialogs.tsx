import { useMemo, useState } from 'react';
import type { SurveySpec } from '../../api/client';

function isOtherOption(opt: string): boolean {
  return /其他|手动输入|其它/.test(opt);
}

interface SurveyProps {
  survey: SurveySpec;
  onSubmit: (answers: Record<string, string>) => void;
  onCancel: () => void;
}

function questionLabel(q: SurveySpec['questions'][0], index: number, total: number): string {
  if (q.step_label) {
    return `${q.step_label}（${index + 1}/${total}）`;
  }
  return `问题 ${index + 1}/${total}`;
}

interface MultiSelectProps {
  questionId: string;
  options: string[];
  optionHints?: string[];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
}

function SurveyMultiSelectPanel({
  questionId,
  options,
  optionHints,
  selected,
  onChange,
}: MultiSelectProps) {
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) {
      return options.map((opt, i) => ({ opt, i }));
    }
    return options
      .map((opt, i) => ({ opt, i }))
      .filter(({ opt, i }) => {
        const hint = optionHints?.[i] || '';
        return opt.toLowerCase().includes(q) || hint.toLowerCase().includes(q);
      });
  }, [options, optionHints, query]);

  const toggle = (opt: string, checked: boolean) => {
    const next = new Set(selected);
    if (checked) {
      next.add(opt);
    } else {
      next.delete(opt);
    }
    onChange(next);
  };

  const selectAll = () => {
    const next = new Set(selected);
    for (const { opt } of filtered) {
      next.add(opt);
    }
    onChange(next);
  };

  const clearAll = () => onChange(new Set());

  const selectOnlyFiltered = () => {
    onChange(new Set(filtered.map(({ opt }) => opt)));
  };

  const hasFilter = query.trim().length > 0;

  return (
    <div className="survey-multi" data-question={questionId}>
      <div className="survey-multi-toolbar">
        <input
          type="search"
          className="survey-multi-search"
          placeholder="搜索项目名或说明…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="survey-multi-count">
          已选 <strong>{selected.size}</strong> / {options.length}
        </span>
      </div>
      <div className="survey-multi-actions">
        <button type="button" className="survey-multi-action" onClick={selectAll}>
          {hasFilter ? '全选筛选结果' : '全选'}
        </button>
        {hasFilter && (
          <button type="button" className="survey-multi-action" onClick={selectOnlyFiltered}>
            仅选筛选
          </button>
        )}
        <button
          type="button"
          className="survey-multi-action"
          onClick={clearAll}
          disabled={selected.size === 0}
        >
          清空
        </button>
      </div>
      <div className="survey-multi-list" role="listbox" aria-multiselectable>
        {filtered.length === 0 ? (
          <p className="survey-multi-empty">无匹配项</p>
        ) : (
          filtered.map(({ opt, i }) => {
            const checked = selected.has(opt);
            const hint = optionHints?.[i];
            return (
              <label
                key={opt}
                className={`survey-multi-item${checked ? ' is-checked' : ''}`}
                title={hint ? `${opt} — ${hint}` : opt}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={(e) => toggle(opt, e.target.checked)}
                />
                <span className="survey-multi-item-main">{opt}</span>
                {hint && <span className="survey-multi-item-hint">{hint}</span>}
              </label>
            );
          })
        )}
      </div>
    </div>
  );
}

export default function SurveyDialog({ survey, onSubmit, onCancel }: SurveyProps) {
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    for (const q of survey.questions) {
      if (q.multi_select) {
        const idx = q.default_indices?.[0] ?? q.default_index ?? 0;
        init[q.id] = q.options[idx] || q.options[0] || '';
      } else {
        init[q.id] = q.options[q.default_index] || q.options[0] || '';
      }
    }
    return init;
  });
  const [multiSelected, setMultiSelected] = useState<Record<string, Set<string>>>(() => {
    const init: Record<string, Set<string>> = {};
    for (const q of survey.questions) {
      if (q.multi_select) {
        const set = new Set<string>();
        const indices = q.default_indices?.length ? q.default_indices : [q.default_index ?? 0];
        for (const i of indices) {
          if (q.options[i]) {
            set.add(q.options[i]);
          }
        }
        init[q.id] = set;
      }
    }
    return init;
  });
  const [freeText, setFreeText] = useState<Record<string, string>>({});

  const total = survey.questions.length;
  const q = survey.questions[step];
  const isLast = step >= total - 1;

  const currentAnswer = useMemo(() => {
    if (!q) {
      return '';
    }
    if (q.multi_select) {
      return Array.from(multiSelected[q.id] || []).join('；');
    }
    let val = answers[q.id] || '';
    if (isOtherOption(val) && freeText[q.id]) {
      val = `${val}：${freeText[q.id]}`;
    }
    return val;
  }, [q, answers, multiSelected, freeText]);

  const canNext = useMemo(() => {
    if (!q) {
      return false;
    }
    if (q.multi_select) {
      return (multiSelected[q.id]?.size || 0) > 0;
    }
    const val = answers[q.id] || '';
    if (!val.trim()) {
      return false;
    }
    if (isOtherOption(val) && q.allow_free_text) {
      return Boolean(freeText[q.id]?.trim());
    }
    return true;
  }, [q, answers, multiSelected, freeText]);

  const commitAndNext = () => {
    if (!q) {
      return;
    }
    const nextAnswers = { ...answers };
    if (q.multi_select) {
      nextAnswers[q.id] = Array.from(multiSelected[q.id] || []).join('；');
    } else if (isOtherOption(answers[q.id] || '') && freeText[q.id]) {
      nextAnswers[q.id] = `${answers[q.id]}：${freeText[q.id]}`;
    }
    setAnswers(nextAnswers);

    if (isLast) {
      onSubmit(nextAnswers);
      return;
    }
    setStep((s) => s + 1);
  };

  if (!q) {
    return null;
  }

  return (
    <div className="modal-overlay survey-overlay">
      <div
        className={`modal survey-modal survey-wizard${q.multi_select ? ' survey-wizard--multi' : ''}`}
        role="dialog"
      >
        <header className="survey-wizard-header">
          <h2>{survey.title}</h2>
          <p className="survey-wizard-progress">{questionLabel(q, step, total)}</p>
        </header>

        <div className="survey-wizard-body">
          <div className="survey-q">
            <label className="survey-wizard-prompt">{q.prompt}</label>

            {q.multi_select ? (
              <SurveyMultiSelectPanel
                questionId={q.id}
                options={q.options}
                optionHints={q.option_hints}
                selected={multiSelected[q.id] || new Set()}
                onChange={(next) => {
                  setMultiSelected((prev) => ({ ...prev, [q.id]: next }));
                }}
              />
            ) : q.allow_free_text && q.options.length <= 1 ? (
            <textarea
              className="survey-free-text"
              rows={4}
              value={freeText[q.id] || answers[q.id] || ''}
              onChange={(e) => {
                setFreeText((prev) => ({ ...prev, [q.id]: e.target.value }));
                setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }));
              }}
            />
          ) : (
            <div className="survey-options">
              {q.options.map((opt, i) => {
                const hint = q.option_hints?.[i];
                const selected = answers[q.id] === opt;
                return (
                  <label key={opt} className={`survey-option-row${selected ? ' is-selected' : ''}`}>
                    <input
                      type="radio"
                      name={q.id}
                      checked={selected}
                      onChange={() => setAnswers((prev) => ({ ...prev, [q.id]: opt }))}
                    />
                    <span className="survey-option-text">{opt}</span>
                    {hint && <span className="survey-option-hint">{hint}</span>}
                  </label>
                );
              })}
            </div>
          )}

          {!q.multi_select && isOtherOption(answers[q.id] || '') && q.allow_free_text && (
            <textarea
              className="survey-free-text survey-other-input"
              rows={2}
              placeholder="请补充说明…"
              value={freeText[q.id] || ''}
              onChange={(e) => setFreeText((prev) => ({ ...prev, [q.id]: e.target.value }))}
            />
          )}
          </div>

          {isLast && (
            <div className="survey-summary">
              <p className="survey-summary-title">确认你的选择</p>
              <ul>
                {survey.questions.map((item, i) => (
                  <li key={item.id}>
                    <span className="survey-summary-q">{item.step_label || item.id}</span>
                    <span className="survey-summary-a">
                      {i === step ? currentAnswer : answers[item.id] || '—'}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="modal-actions survey-wizard-actions">
          <button type="button" onClick={onCancel}>
            取消
          </button>
          {step > 0 && (
            <button type="button" onClick={() => setStep((s) => s - 1)}>
              上一步
            </button>
          )}
          <button
            type="button"
            className="primary"
            disabled={!canNext}
            onClick={commitAndNext}
          >
            {isLast ? '提交确认' : '下一步'}
          </button>
        </div>
      </div>
    </div>
  );
}

interface PlanConfirmProps {
  payload: Record<string, unknown>;
  onConfirm: (action: string, allowWrite: boolean, reviseNote: string) => void;
  onCancel: () => void;
}

export function PlanConfirmDialog({ payload, onConfirm, onCancel }: PlanConfirmProps) {
  const [step, setStep] = useState(0);
  const [action, setAction] = useState<'approve' | 'revise' | 'cancel'>('approve');
  const [allowWrite, setAllowWrite] = useState(false);
  const [reviseNote, setReviseNote] = useState('');
  const tasks = (payload.tasks as Array<{ id: string; title: string }>) || [];
  const title = String(payload.title || '未命名计划');

  const finish = () => {
    if (action === 'cancel') {
      onConfirm('cancel', false, '');
      return;
    }
    if (action === 'revise') {
      onConfirm('revise', allowWrite, reviseNote);
      return;
    }
    onConfirm('approve', allowWrite, '');
  };

  return (
    <div
      className="modal-overlay plan-confirm-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onCancel();
        }
      }}
    >
      <div className="plan-confirm-modal survey-wizard" role="dialog" aria-labelledby="plan-confirm-title">
        {step === 0 ? (
          <>
            <header className="plan-confirm-header">
              <span className="plan-confirm-badge">待确认 · 1/2</span>
              <h2 id="plan-confirm-title" className="plan-confirm-title">
                {title}
              </h2>
              <p className="plan-confirm-meta">{tasks.length} 个 Work 任务</p>
            </header>
            <div className="plan-confirm-body">
              <ul className="plan-confirm-tasks">
                {tasks.map((t) => (
                  <li key={t.id} className="plan-confirm-task">
                    <span className="plan-confirm-task-id">{t.id}</span>
                    <span className="plan-confirm-task-title">{t.title}</span>
                  </li>
                ))}
              </ul>
              <div className="survey-options plan-confirm-action-options">
                {(
                  [
                    ['approve', '确认并开始执行'],
                    ['revise', '修改计划（/plan revise）'],
                    ['cancel', '取消'],
                  ] as const
                ).map(([val, label]) => (
                  <label
                    key={val}
                    className={`survey-option-row${action === val ? ' is-selected' : ''}`}
                  >
                    <input
                      type="radio"
                      name="plan-action"
                      checked={action === val}
                      onChange={() => setAction(val)}
                    />
                    <span className="survey-option-text">{label}</span>
                  </label>
                ))}
              </div>
              {action === 'revise' && (
                <div className="plan-confirm-revise">
                  <label htmlFor="plan-revise-note" className="plan-confirm-revise-label">
                    修订说明
                  </label>
                  <textarea
                    id="plan-revise-note"
                    className="plan-confirm-revise-input"
                    placeholder="例如：合并 w3/w4，增加性能测试项…"
                    rows={3}
                    value={reviseNote}
                    onChange={(e) => setReviseNote(e.target.value)}
                  />
                </div>
              )}
            </div>
            <footer className="plan-confirm-footer">
              <button type="button" className="plan-confirm-btn plan-confirm-btn-ghost" onClick={onCancel}>
                关闭
              </button>
              <button
                type="button"
                className="plan-confirm-btn plan-confirm-btn-primary"
                onClick={() => {
                  if (action === 'cancel') {
                    finish();
                    return;
                  }
                  setStep(1);
                }}
              >
                下一步
              </button>
            </footer>
          </>
        ) : (
          <>
            <header className="plan-confirm-header">
              <span className="plan-confirm-badge">写权限 · 2/2</span>
              <h2 className="plan-confirm-title">允许 Worker 写文件？</h2>
            </header>
            <div className="plan-confirm-body">
              <div className="survey-options">
                {(['否（只读）', '是（受 task scope 限制）'] as const).map((opt) => {
                  const yes = opt.startsWith('是');
                  return (
                    <label
                      key={opt}
                      className={`survey-option-row${allowWrite === yes ? ' is-selected' : ''}`}
                    >
                      <input
                        type="radio"
                        name="plan-write"
                        checked={allowWrite === yes}
                        onChange={() => setAllowWrite(yes)}
                      />
                      <span className="survey-option-text">{opt}</span>
                    </label>
                  );
                })}
              </div>
            </div>
            <footer className="plan-confirm-footer">
              <button type="button" className="plan-confirm-btn plan-confirm-btn-ghost" onClick={() => setStep(0)}>
                上一步
              </button>
              <button
                type="button"
                className="plan-confirm-btn plan-confirm-btn-primary"
                onClick={finish}
              >
                {action === 'approve' ? '批准执行' : action === 'revise' ? '提交修订' : '确认'}
              </button>
            </footer>
          </>
        )}
      </div>
    </div>
  );
}

interface TaskStepProps {
  taskId: string;
  onContinue: () => void;
  onDismiss: () => void;
}

export function TaskStepConfirmDialog({ taskId, onContinue, onDismiss }: TaskStepProps) {
  return (
    <div className="modal-overlay">
      <div className="modal task-step-modal" role="dialog">
        <h2>Work {taskId} 已完成</h2>
        <p>是否继续执行下一批任务？（等同终端空 Enter / Continue）</p>
        <div className="modal-actions">
          <button type="button" onClick={onDismiss}>
            稍后
          </button>
          <button type="button" className="primary" onClick={onContinue}>
            继续执行
          </button>
        </div>
      </div>
    </div>
  );
}
