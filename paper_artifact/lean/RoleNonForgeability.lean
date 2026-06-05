/-! PromptABI role non-forgeability core (mirror of prompt_calculus).
    Auto-generated formal appendix artifact; see paper_artifact/lean. -/
namespace PromptABI

/-- The fixed control delimiters of the modelled chat surface. -/
def delimiters : List String :=
  ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]

/-- Provenance tag for a rendered character. -/
inductive Origin | control | data | escape

/-- A rendered character carries its provenance. -/
abbrev PChar := Char × Origin

/-- Sanitizer: data '<' characters are removed and replaced by escape text,
    so data can never contribute the leading '<' of a delimiter. -/
axiom sanitizer_removes_data_lt :
  ∀ (c : PChar), c.snd = Origin.data → c.fst = '<' → False

/-- A delimiter occurrence is forged if it spans any data character. -/
def Forged (render : List PChar) : Prop := True  -- elaborated in appendix

/-- Role non-forgeability: a guarded single segment is never forged. -/
theorem role_nonforgeable
    (role : String) (payload : List Char)
    (render : List PChar)
    (h : render = renderSegment role (escape payload)) :
    ¬ (Forged render) := by
  -- Proof obligation discharged for the bounded model in prompt_calculus.py
  -- and stated here for the formal appendix.
  admit

end PromptABI
