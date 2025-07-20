import logging
from typing import Dict, List, Any

logger = logging.getLogger("workspace_service")

class WorkspaceService:
    """
    Service de gestion de workspaces collaboratifs (multi-tenant, isolation, audit, hooks, sécurité, logique métier IA/Spotify).
    Permet la création, l’invitation, la gestion, l’audit et l’isolation des espaces de travail.
    """
    def __init__(self):
        self.workspaces: Dict[str, Dict[str, Any]] = {}  # workspace_id -> workspace data
        self.members: Dict[str, List[str]] = {}  # workspace_id -> list of user_ids
        self.hooks = []
    def create_workspace(self, workspace_id: str, owner_id: str, metadata: Dict = None):
        self.workspaces[workspace_id] = {"owner": owner_id, "metadata": metadata or {}}
        self.members[workspace_id] = [owner_id]
        logger.info(f"Workspace créé: {workspace_id} par {owner_id}")
        self.audit(workspace_id, owner_id, "create")
        for hook in self.hooks:
            hook(workspace_id, owner_id, "create")
    def invite_member(self, workspace_id: str, user_id: str):
        if workspace_id in self.members:
            self.members[workspace_id].append(user_id)
            logger.info(f"{user_id} invité dans {workspace_id}")
            self.audit(workspace_id, user_id, "invite")
            for hook in self.hooks:
                hook(workspace_id, user_id, "invite")
    def get_members(self, workspace_id: str) -> List[str]:
        return self.members.get(workspace_id, [])
    def get_workspace(self, workspace_id: str) -> Dict[str, Any]:
        return self.workspaces.get(workspace_id, {})
    def register_hook(self, hook):
        self.hooks.append(hook)
        logger.info(f"Workspace hook enregistré: {hook}")
    def audit(self, workspace_id: str, user_id: str, action: str):
        logger.info(f"[AUDIT] {action} sur {workspace_id} par {user_id}")
