"""One active identity backend with explicit, non-interchangeable model assets."""
def backend_config(config,name=None):
    selected=name or config.get('backend','sface')
    if selected not in ('sface','seeta6'):raise ValueError('Unknown familiar-face backend')
    result={k:v for k,v in config.items() if k!='backends'}
    if 'backends' in config:
        if selected not in config['backends']:raise ValueError('Missing familiar-face backend configuration')
        result.update(config['backends'][selected])
    result['backend']=selected
    result['score_threshold']=result.get('score_threshold',result.get('cosine_threshold',.5))
    return result
