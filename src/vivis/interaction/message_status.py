"""Canonical delivery facts and trusted receipt text; no audio or device code."""

DELIVERY={
    'pending':('已保存，等待转告','等待收件人出现并转告'),
    'announced':('已转告，等待确认收到','等待收件人明确确认收到留言'),
    'completed':('收件人已确认收到','收件人已确认收到留言；不跟踪所述事项的实际执行'),
    'cancelled':('已取消，不再提醒','转告已取消，不再自动提醒'),
}

RECEIPTS={
    'created':'这条留言已保存，我会在确认收件人身份后转告。',
    'merged':'这条留言已合并更新，不会重复建单。',
    'unchanged':'这条留言已经记录，不会重复建单。',
    'already_acknowledged':'这条留言之前已确认送达，不会再次提醒。',
    'acknowledged':'已记录您收到这条留言，不会再重复提醒。',
    'cancelled':'这条留言已取消。',
    'failed':'这次留言操作失败了，请稍后查询或重试。',
    'unconfirmed':'这次留言操作结果尚未确认，请先查询留言状态，避免重复记录。',
    'clarify':'这是补充之前的留言，还是另外一条新留言？',
    'meaning':'留言里的关键内容没有听清，请补充没有听清的部分。这条留言还没有保存。',
    'recipient':'这条留言是给谁的？请告诉我收件人。这条留言还没有保存。',
    'unsupported_recipient':'目前还不能自动向未登记的访客转告，这条留言没有保存。',
    'delivery_scope':'我只负责保存和转告留言。您想给谁留什么话？',
    'not_saved':'这条留言还没有保存成功，您可以查询状态或要求重试。',
    'source_timeout':'本轮识别文字暂未返回，这条留言尚未保存。',
}


def message_view(record):
    """Project persisted truth without rewriting legacy records or source text."""
    result=dict(record)
    if result.get('kind') not in ('message','reminder'):return result
    status='pending' if result['status']=='in_progress' else result['status']
    label,next_action=DELIVERY[status]
    result.update(saved=True,delivery_status=status,delivery_label=label,
        delivery_summary=label+'；不代表留言所述事情已实际办完。',
        next_action=next_action,physical_execution_tracked=False)
    return result


def operation_failure(status,error):
    """An unavailable readback is uncertainty, not evidence of a failed commit."""
    unconfirmed=status=='database_operation_failed'
    return dict(operation_status='unconfirmed' if unconfirmed else
        'waiting' if status in ('source_transcript_pending','semantic_reconciliation_required') else
        'needs_clarification' if status in ('meaning_clarification_required','recipient_unavailable','identity_confirmation_required') else 'rejected',
        saved=None if unconfirmed else False,error=error)
